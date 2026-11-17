"""GAM probit à P-splines : second modèle additif du blend final.

Le bruit du générateur de cibles est gaussien (experiments.md, 2026-10-10), donc
P(y=1|x) = Φ(b + Σ_j f_j(x_j)). LightGBM estime chaque f_j en escalier ; ce modèle
les estime par des courbes lisses à base de splines, ce qui donne des erreurs
différentes et améliore le blend.

Pour chaque feature : transformation par le rang (uniformise les ratios très
asymétriques), B-splines cubiques centrées à n_knots nœuds, pénalité λ sur les
différences secondes des coefficients (P-splines d'Eilers et Marx), monotonie
optionnelle par coefficients de même signe (mêmes sens que LightGBM).

Déterministe (L-BFGS-B à gradient exact, pas de seed), mais les produits matriciels
tournent sur un seul thread : la vraisemblance est plate le long de certaines
directions, et sans ça le point d'arrêt de L-BFGS-B dépendait du nombre de cœurs de
la machine (écarts jusqu'à 8e-4 sur une probabilité).
"""

import numpy as np
from scipy import optimize, special
from scipy.interpolate import BSpline
from threadpoolctl import threadpool_limits

SPLINE_DEGREE = 3


def _rank_transform(train_sorted, x):
    """Rang moyen de x parmi les valeurs du train (ex aequo compris), ramené à [0, 1]."""
    below = np.searchsorted(train_sorted, x, side="left")
    below_or_equal = np.searchsorted(train_sorted, x, side="right")
    return (below + below_or_equal) / (2 * len(train_sorted))


class ProbitGAM:
    """Modèle additif probit à P-splines, avec contraintes de monotonie optionnelles.

    `monotone` : liste alignée sur les colonnes, 1 (croissante), -1 (décroissante) ou
    0 (libre).
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
        # Poids d'échantillon (pseudo-labeling, 2026-10-15) : étiquette douce y∈[0,1] et
        # poids w, taille effective n_eff=Σw à la place de n pour garder la force de la
        # pénalité λ comparable (sinon noyée par les lignes de test ajoutées).
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        n_eff = w.sum()
        n, p = X.shape
        n_basis = self.n_knots + SPLINE_DEGREE + 1
        self.train_sorted_ = [np.sort(X[:, j]) for j in range(p)]
        self.basis_means_ = [self._raw_basis(j, X[:, j]).mean(axis=0) for j in range(p)]

        # Coefficients de la feature j : beta_j = [0, cumsum(delta_j)]. La courbe est
        # monotone si tous les incréments delta_j ont le même signe (bornes L-BFGS-B).
        # Comme la base est centrée, le premier coefficient est fixé à 0 sans perte.
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
        """Score η sur l'échelle probit."""
        X = np.asarray(X, dtype=float)
        eta = np.full(len(X), self.intercept_)
        with threadpool_limits(limits=1, user_api="blas"):
            for j, beta in enumerate(self.coef_):
                eta += self._basis(j, X[:, j]) @ beta
        return eta

    def predict_proba(self, X):
        p = special.ndtr(self.decision_function(X))
        return np.column_stack([1 - p, p])
