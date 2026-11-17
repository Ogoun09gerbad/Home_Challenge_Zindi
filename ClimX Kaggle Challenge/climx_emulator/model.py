"""Forced-analogue emulator: daily fields from forcings only.

1. A small energy-balance regression turns GHG and aerosol forcings into a forced global
   temperature for any scenario year.
2. Per cell, month and variable, a regression gives the forced monthly mean as a function of
   that forced state.
3. A realisation of a target year copies a whole library year with a similar forced state,
   then shifts it by the difference between the target and library forced means. Daily weather,
   spells, tails, and consistency across cells and variables come from the library year.

Averaging indices over realisations estimates their expected value given the forcing, which
is what minimises the squared error in the metric.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from common import DAYS, TARGETS, Run

MULTIPLICATIVE = {"pr", "huss", "sfcWind"}
PR_FLOOR = 1e-9
MONTH_LENGTHS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
MONTH_OF_DAY = np.repeat(np.arange(12), MONTH_LENGTHS)
MONTH_CENTERS = np.cumsum(MONTH_LENGTHS) - MONTH_LENGTHS / 2


def _monthly_to_daily(monthly: np.ndarray) -> np.ndarray:
    """Periodic linear interpolation of a (12, ...) array to (365, ...)."""
    days = np.arange(DAYS) + 0.5
    x = np.concatenate([[MONTH_CENTERS[-1] - DAYS], MONTH_CENTERS, [MONTH_CENTERS[0] + DAYS]])
    pos = np.clip(np.searchsorted(x, days) - 1, 0, len(x) - 2)
    frac = ((days - x[pos]) / (x[pos + 1] - x[pos])).reshape((-1,) + (1,) * (monthly.ndim - 1))
    extended = np.concatenate([monthly[-1:], monthly, monthly[:1]])
    return extended[pos] * (1 - frac) + extended[pos + 1] * frac


def _ghg_forcing(run: Run, reference: Run) -> np.ndarray:
    """Simplified radiative forcing (W m-2), relative to the reference's start.

    The dataset stores mole fractions; the CH4 and N2O coefficients expect ppb.
    """
    co2, ch4, n2o = (run.ghg[v] for v in ("CO2_LBC", "CH4_LBC", "N2O_LBC"))
    co2_0, ch4_0, n2o_0 = (reference.ghg[v][0] for v in ("CO2_LBC", "CH4_LBC", "N2O_LBC"))
    return (
        5.35 * np.log(co2 / co2_0)
        + 0.036 * (np.sqrt(ch4 * 1e9) - np.sqrt(ch4_0 * 1e9))
        + 0.12 * (np.sqrt(n2o * 1e9) - np.sqrt(n2o_0 * 1e9))
    )


def _lagged(series: np.ndarray, tau: float, start: float = 0.0) -> np.ndarray:
    """First-order low-pass filter: the response of a heat reservoir with time scale tau."""
    out = np.empty_like(series)
    state = start
    k = 1.0 / tau
    for i, value in enumerate(series):
        state = state + k * (value - state)
        out[i] = state
    return out


@dataclass
class Config:
    n_analogues: int = 48  # library years a target year can draw from
    taus: tuple[float, ...] = (4.0, 60.0)  # response time scales of the energy-balance fit (years)
    use_aerosols: bool = True
    aerosol_weight: float = 0.5  # weight of aerosol distance in analogue selection
    quadratic: bool = True  # allow forced means to curve with warming
    apply_shift: bool = True  # False: copy library years without adjusting them
    seed: int = 0


@dataclass
class ForcedAnalogueEmulator:
    config: Config = field(default_factory=Config)

    def _state(self, hist: Run, run: Run | None = None) -> np.ndarray:
        """Forcing predictors of a run, continued from the end of the historical run if a scenario."""
        f_hist = _ghg_forcing(hist, hist)
        a_hist = np.log(self._aerosol(hist))
        if run is None:
            f, a = f_hist, a_hist
            n = 0
        else:
            f = np.concatenate([f_hist, _ghg_forcing(run, hist)])
            a = np.concatenate([a_hist, np.log(self._aerosol(run))])
            n = len(f_hist)
        cols = [_lagged(f, tau)[n:] for tau in self.config.taus]
        cols.append(_lagged(a, 2.0, start=a[0])[n:])
        return np.column_stack(cols)

    def _aerosol(self, run: Run) -> np.ndarray:
        # SO2 dominates the aerosol cooling; BC_AX and OM_NI are kept out of this first version.
        return np.maximum(run.aerosols["SO2"], 1e-30)

    def fit(self, hist: Run, library: list[Run], lat: np.ndarray) -> "ForcedAnalogueEmulator":
        self.hist = hist
        self.library = [hist] + library
        cos = np.cos(np.deg2rad(lat))
        self.weights = cos / cos.sum()

        # Forced global temperature from all runs (energy-balance regression).
        states = [self._state(hist)] + [self._state(hist, r) for r in library]
        self.lib_runs = np.concatenate([np.full(len(r.years), i) for i, r in enumerate(self.library)])
        self.lib_years = np.concatenate([np.arange(len(r.years)) for r in self.library])
        s = np.concatenate(states)
        if not self.config.use_aerosols:
            s = s[:, :-1]
        self.state_mean, self.state_std = s.mean(0), s.std(0)
        gmst = np.concatenate([self._gmst(r) for r in self.library])
        design = self._design_state(s)
        self.ebm, *_ = np.linalg.lstsq(design, gmst, rcond=None)
        self.lib_forced = design @ self.ebm
        self.lib_aero = np.concatenate(states)[:, -1]
        self.aero_scale = self.lib_aero.std() or 1.0
        self.ebm_rmse = float(np.sqrt(np.mean((gmst - self.lib_forced) ** 2)))

        # Forced monthly means, per cell and variable, as a function of the forced state.
        design_means = self._design_means(self.lib_forced, self.lib_aero)
        self.coef = {}
        for var in TARGETS:
            monthly = np.concatenate([self._monthly(r.fields[var], var) for r in self.library])
            n, months, nlat, nlon = monthly.shape
            flat = monthly.reshape(n, -1)
            coef, *_ = np.linalg.lstsq(design_means, flat, rcond=None)
            self.coef[var] = coef.reshape((design_means.shape[1], months, nlat, nlon)).astype(np.float64)
        return self

    def _gmst(self, run: Run) -> np.ndarray:
        annual = run.fields["tas"].mean(axis=1)
        return (annual.mean(axis=2) * self.weights).sum(axis=1)

    def _design_state(self, s: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(len(s)), (s - self.state_mean) / self.state_std])

    def _design_means(self, forced: np.ndarray, aero: np.ndarray) -> np.ndarray:
        t = forced - self.lib_forced_ref()
        cols = [np.ones_like(t), t]
        if self.config.quadratic:
            cols.append(t**2)
        if self.config.use_aerosols:
            cols.append((aero - self.lib_aero.mean()) / self.aero_scale)
        return np.column_stack(cols)

    def lib_forced_ref(self) -> float:
        return float(self.lib_forced.mean())

    def _monthly(self, block: np.ndarray, var: str) -> np.ndarray:
        """(year, 365, lat, lon) -> (year, 12, lat, lon) monthly means, log for positive variables."""
        out = np.stack([block[:, MONTH_OF_DAY == m].mean(axis=1) for m in range(12)], axis=1)
        return np.log(np.maximum(out, PR_FLOOR)) if var in MULTIPLICATIVE else out

    def forced_state(self, run: Run) -> tuple[np.ndarray, np.ndarray]:
        s = self._state(self.hist, run)
        aero = s[:, -1]
        if not self.config.use_aerosols:
            s = s[:, :-1]
        return self._design_state(s) @ self.ebm, aero

    def _forced_daily(self, forced: float, aero: float, var: str) -> np.ndarray:
        """Forced mean of one year as a (365, lat, lon) array, in the variable's regression space."""
        d = self._design_means(np.array([forced]), np.array([aero]))[0]
        return _monthly_to_daily(np.tensordot(d, self.coef[var], axes=(0, 0)))

    def choose_analogues(self, forced: np.ndarray, aero: np.ndarray, available: np.ndarray | None = None) -> list[np.ndarray]:
        """For each target year, the library entries closest in forced state, among those available."""
        chosen = []
        for f, a in zip(forced, aero):
            dist = np.abs(self.lib_forced - f)
            if self.config.use_aerosols:
                dist = dist + self.config.aerosol_weight * np.abs(self.lib_aero - a) / self.aero_scale
            if available is not None:
                dist = np.where(available, dist, np.inf)
            chosen.append(np.argsort(dist, kind="stable")[: self.config.n_analogues])
        return chosen

    def realization(self, run: Run, r: int, variables: list[str] = TARGETS) -> dict[str, np.ndarray]:
        """One daily realisation of a scenario as var -> (year, 365, lat, lon)."""
        forced, aero = self.forced_state(run)
        chosen = self.choose_analogues(forced, aero)
        out = {v: None for v in variables}
        for y, candidates in enumerate(chosen):
            # A per-year permutation so that realisations use distinct library years.
            order = np.random.default_rng([self.config.seed, y]).permutation(len(candidates))
            entry = candidates[order[r % len(candidates)]]
            source = self.library[self.lib_runs[entry]]
            year = self.lib_years[entry]
            for var in variables:
                daily = source.fields[var][year].astype(np.float64)
                if self.config.apply_shift:
                    delta = self._forced_daily(forced[y], aero[y], var) - self._forced_daily(
                        self.lib_forced[entry], self.lib_aero[entry], var
                    )
                    daily = daily * np.exp(delta) if var in MULTIPLICATIVE else daily + delta
                if out[var] is None:
                    out[var] = np.empty((len(forced),) + daily.shape, dtype=np.float32)
                out[var][y] = daily
        # Shifts are per variable: restore tasmin <= tas <= tasmax.
        if {"tas", "tasmax", "tasmin"} <= set(variables):
            out["tasmax"] = np.maximum(out["tasmax"], out["tas"])
            out["tasmin"] = np.minimum(out["tasmin"], out["tas"])
        return out
