"""Probit P-spline GAM: second additive model of the final blend.

The noise of the target generator is Gaussian (experiments.md, 2026-10-10), so
P(y=1|x) = Φ(b + Σ_j f_j(x_j)). LightGBM estimates each f_j as a step function; this
model estimates them as smooth spline-based curves, which gives different errors
and improves the blend.

For each feature: rank transformation (uniformises highly skewed
ratios), centred cubic B-splines with n_knots knots, penalty λ on the
second differences of the coefficients (Eilers and Marx P-splines), optional
monotonicity via same-sign coefficients (same directions as LightGBM).

Deterministic (L-BFGS-B with exact gradient, no seed), but the matrix products
run on a single thread: the likelihood is flat along some
directions, and without this the L-BFGS-B stopping point depended on the machine's
number of cores (differences of up to 8e-4 on a probability).
"""

import numpy as np
from scipy import optimize, special
from scipy.interpolate import BSpline
from threadpoolctl import threadpool_limits

SPLINE_DEGREE = 3


def _rank_transform(train_sorted, x):
    """Mean rank of x among the train values (ties included), scaled to [0, 1]."""
    below = np.searchsorted(train_sorted, x, side="left")
    below_or_equal = np.searchsorted(train_sorted, x, side="right")
    return (below + below_or_equal) / (2 * len(train_sorted))


class ProbitGAM:
    """Additive probit P-spline model, with optional monotonicity constraints.

    `monotone`: list aligned with the columns, 1 (increasing), -1 (decreasing) or
    0 (free).
    """

    def __init__(self, n_knots=20, lam=5.0, monotone=None, max_iter=5000):
        self.n_knots = n_knots
        self.lam = lam
        self.monotone = monotone
        self.max_iter = max_iter

    def _raw_basis(self, j, x):
        u = np.clip(_rank_transform(self.train_sorted_[j], x), 0.0, 1.0 - 1e-12)
        knots = np.r_[[0.0] * SPLINE_DEGREE, np.linspace(0, 1, self.n_knots + 2), [1.0] * SPLINE_DEGREE]
        return BSpline.design_matrix(u, knots, SPLINE_DEGREE).toarray()

    def _basis(self, j, x):
        return self._raw_basis(j, x) - self.basis_means_[j]

    def fit(self, X, y, sample_weight=None):
        with threadpool_limits(limits=1, user_api="blas"):
            return self._fit(X, y, sample_weight)

    def _fit(self, X, y, sample_weight=None):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        # Sample weights (pseudo-labeling, 2026-10-15): soft label y∈[0,1] and
        # weight w, effective size n_eff=Σw instead of n to keep the strength of the
        # penalty λ comparable (otherwise drowned out by the added test rows).
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        n_eff = w.sum()
        n, p = X.shape
        n_basis = self.n_knots + SPLINE_DEGREE + 1
        self.train_sorted_ = [np.sort(X[:, j]) for j in range(p)]
        self.basis_means_ = [self._raw_basis(j, X[:, j]).mean(axis=0) for j in range(p)]

        # Coefficients of feature j: beta_j = [0, cumsum(delta_j)]. The curve is
        # monotone if all increments delta_j have the same sign (L-BFGS-B bounds).
        # Since the basis is centred, the first coefficient is fixed to 0 without loss.
        cumsum = np.vstack([np.zeros(n_basis - 1), np.tril(np.ones((n_basis - 1, n_basis - 1)))])
        second_diff = np.diff(np.eye(n_basis), 2, axis=0) @ cumsum
        penalty = second_diff.T @ second_diff
        design = np.hstack([self._basis(j, X[:, j]) @ cumsum for j in range(p)])
        k = n_basis - 1

        bounds = [(None, None)]
        for j in range(p):
            direction = 0 if self.monotone is None else self.monotone[j]
            bounds += [(0, None) if direction > 0 else (None, 0) if direction < 0 else (None, None)] * k

        def objective(theta):
            eta = theta[0] + design @ theta[1:]
            log_cdf, log_sf = special.log_ndtr(eta), special.log_ndtr(-eta)
            log_pdf = -0.5 * eta**2 - 0.5 * np.log(2 * np.pi)
            nll = -np.sum(w * (y * log_cdf + (1 - y) * log_sf))
            grad_eta = w * (-y * np.exp(log_pdf - log_cdf) + (1 - y) * np.exp(log_pdf - log_sf))
            deltas = theta[1:].reshape(p, k)
            pen_grad = self.lam * deltas @ penalty
            value = nll + 0.5 * np.sum(pen_grad * deltas)
            grad = np.r_[grad_eta.sum(), design.T @ grad_eta + pen_grad.ravel()]
            return value / n_eff, grad / n_eff

        theta0 = np.r_[special.ndtri(np.average(y, weights=w)), np.zeros(p * k)]
        result = optimize.minimize(objective, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                                   options={"maxiter": self.max_iter, "maxfun": 2 * self.max_iter})
        self.intercept_ = result.x[0]
        self.coef_ = [cumsum @ d for d in result.x[1:].reshape(p, k)]
        self.n_iter_ = result.nit
        return self

    def decision_function(self, X):
        """Score η on the probit scale."""
        X = np.asarray(X, dtype=float)
        eta = np.full(len(X), self.intercept_)
        with threadpool_limits(limits=1, user_api="blas"):
            for j, beta in enumerate(self.coef_):
                eta += self._basis(j, X[:, j]) @ beta
        return eta

    def predict_proba(self, X):
        p = special.ndtr(self.decision_function(X))
        return np.column_stack([1 - p, p])
