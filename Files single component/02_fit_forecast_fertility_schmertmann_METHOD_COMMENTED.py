# =============================================================================
# FERTILITY MODULE - ESTIMATION AND FORECAST
# =============================================================================
# Reading guide
# -------------
# This script estimates and forecasts age-specific fertility rates (ASFR).
# The demographic idea is to separate fertility into two parts:
#   1. intensity: how many children women have on average, summarized by TFR;
#   2. timing/shape: at which ages births are concentrated, summarized through
#      Schmertmann's three intuitive parameters alpha, P, and H.
#
# The script therefore does four things:
#   - cleans observed ASFR and TFR data;
#   - fits Schmertmann age-pattern parameters to each observed ASFR schedule;
#   - forecasts TFR and age-pattern parameters with time-series models;
#   - reconstructs future ASFR schedules and simulation intervals.
#
# The comments focus on the demographic reason for each step, not on Python syntax.
# =============================================================================

"""
02_fit_forecast_fertility_schmertmann.py

Full fertility module with Schmertmann quadratic splines:
- fits alpha, P, H to observed ASFR schedules by year and citizenship
- forecasts TFR and alpha/P/H with ARIMA selected by AIC
- simulates future trajectories
- rebuilds future ASFR schedules from projected TFR + projected Schmertmann shape
- exports presentation-ready tables

Expected inputs
---------------
data/processed/asfr.csv
    Required columns (flexible names accepted):
    - year
    - age_mother (or age)
    - citizenship
    - asfr

data/processed/TFR.csv
    Required columns:
    - year
    - citizenship
    - tfr

Outputs
-------
output/fertility/
    - schmertmann_parameter_history.csv
    - schmertmann_parameter_forecast_mean.csv
    - schmertmann_parameter_forecast_simulations.csv
    - tfr_forecast_mean.csv
    - tfr_forecast_simulations.csv
    - fertility_forecast_schedule_mean.csv
    - fertility_forecast_schedule_simulations.csv
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

PROJECT_DIR = Path("/Users/andreaballerini/Downloads/projection_project")
INPUT_DIR = PROJECT_DIR
OUTPUT_DIR = PROJECT_DIR / "output" / "fertility"

ASFR_FILE = INPUT_DIR / "asfr.csv"
TFR_FILE = INPUT_DIR / "TFR.csv"


FORECAST_START = None
FORECAST_END = 2075
N_SIM = 3000
AGE_MIN = 14
AGE_MAX = 50
CITIZENSHIP_KEEP = None
ARIMA_CANDIDATES = [(0,1,0),(1,1,0),(0,1,1),(1,1,1),(2,1,0),(2,0,0),(1,0,0)]
RANDOM_SEED = 123
np.random.seed(RANDOM_SEED)


# ============================================================
# Foreign TFR: gap-over-Italians approach
# ============================================================
ITALIAN_LABEL = "italiani"
FOREIGN_LABEL = "stranieri"

# Italians: keep the order that worked well
ITALIAN_TFR_ORDER = (2, 0, 0)

# Foreigners: forecast the gap over Italians, not the level itself
FOREIGN_GAP_ORDER = (1, 0, 0)

# Minimum positive gap: foreigners should stay above Italians
FOREIGN_MIN_GAP = 0.05

# Soft convergence settings
# 0 = fully target path, 1 = fully ARIMA gap forecast
FOREIGN_GAP_BLEND_START = 0.25
FOREIGN_GAP_BLEND_END = 0.60

# Speed of convergence of the target gap toward the minimum gap
# lower = slower, softer decline
FOREIGN_GAP_DECAY = 0.04

# Small numerical constant
EPS = 1e-6


# # ============================================================
# # Fixed ARIMA orders for TFR, by citizenship
# # ============================================================
# ITALIAN_LABEL = "italiani"
# FOREIGN_LABEL = "stranieri"

# # For alpha, P, H keep a small stable candidate set
# ARIMA_CANDIDATES_PARAMS = [
#     (0, 1, 0),
#     (1, 1, 0),
#     (1, 0, 0),
#     (2, 0, 0),
# ]

# # ============================================================
# # TFR constraints for foreigners
# # ============================================================
# ITALIAN_LABEL = "italiani"
# FOREIGN_LABEL = "stranieri"

# # If True, damp the foreign TFR path toward the last observed foreign TFR
# DAMP_FOREIGN_TFR = True

# # Weight applied to ARIMA deviation from the last observed foreign TFR.
# # 0 = fully flat at last observed value
# # 1 = full ARIMA forecast
# # This will increase gradually over the forecast horizon.
# FOREIGN_DAMP_START = 0.20
# FOREIGN_DAMP_END = 0.65

# # Foreign TFR cannot go below Italian TFR
# ENFORCE_FOREIGN_ABOVE_ITALIAN = True

# # Optional extra lower bound for foreigners
# # Set to None if you do not want this extra floor
# FOREIGN_TFR_ABSOLUTE_FLOOR = 1.00

# ============================================================
# ARIMA settings for alpha, P, H
# ============================================================
ARIMA_CANDIDATES_PARAMS = [
    (0, 1, 0),
    (1, 1, 0),
    (1, 0, 0),
    (2, 0, 0),
]

# ============================================================
# Soft constraints for Schmertmann parameters
# ============================================================
ALPHA_MIN = 14.5
ALPHA_MAX = 24.0

P_MIN = 24.0
P_MAX = 37.5

H_MIN = 28.0
H_MAX = 45.0

MIN_P_MINUS_ALPHA = 4.0
MIN_H_MINUS_P = 2.0

USE_PARAM_SHRINKAGE = True
PARAM_TARGET_WINDOW = 5
PARAM_BLEND_START = 0.35
PARAM_BLEND_END = 0.65

def _normalize_colname(s: str) -> str:
    """
    Normalize column names so that small differences in accents, spaces, and capitalization
    do not break the import step.
    """
    s = s.strip().lower()
    for a,b in [("à","a"),("é","e"),("ì","i"),("ò","o"),("ù","u")]:
        s = s.replace(a,b)
    for x in ["(",")","/","-"," "]:
        s = s.replace(x,"_")
    while "__" in s:
        s = s.replace("__","_")
    return s


def _rename_with_aliases(df: pd.DataFrame, aliases: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Rename input columns to the names used internally by the pipeline. This lets the code
    accept Italian/English labels while keeping the rest of the script stable.
    """
    cols = {_normalize_colname(c): c for c in df.columns}
    rename = {}
    for target, options in aliases.items():
        for opt in options:
            key = _normalize_colname(opt)
            if key in cols:
                rename[cols[key]] = target
                break
    return df.rename(columns=rename)


def _load_csv(path: Path) -> pd.DataFrame:
    """
    Read a CSV file with a few common encodings/separators. Italian statistical files often
    vary in separator and encoding, so this avoids manual editing before analysis.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    for sep in [",",";","\t"]:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    return pd.read_csv(path)


def load_asfr(path: Path) -> pd.DataFrame:
    """
    Load observed age-specific fertility rates. These are the schedules used to estimate the
    fertility age pattern by year and citizenship.
    """
    df = _load_csv(path)
    df = _rename_with_aliases(df, {
        "year": ["year","anno","time_period"],
        "age_mother": ["age_mother","eta_madre","eta","age"],
        "citizenship": ["citizenship","cittadinanza","citizenship_group"],
        "asfr": ["asfr","tasso_specifico_fecondita","rate","value","osservazione"],
    })
    need = ["year","age_mother","citizenship","asfr"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"ASFR file is missing columns: {miss}")
    df = df[need].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["age_mother"] = (
        df["age_mother"].astype(str)
        .str.replace(" anni e più", "", regex=False)
        .str.replace(" anni", "", regex=False)
        .str.replace("+", "", regex=False)
    )
    df["age_mother"] = pd.to_numeric(df["age_mother"], errors="coerce")
    df["asfr"] = pd.to_numeric(df["asfr"], errors="coerce")
    df["citizenship"] = df["citizenship"].astype(str).str.strip()
    df = df.dropna(subset=["year","age_mother","citizenship","asfr"])
    df = df[(df["age_mother"] >= AGE_MIN) & (df["age_mother"] <= AGE_MAX)]
    if CITIZENSHIP_KEEP is not None:
        df = df[df["citizenship"].isin(CITIZENSHIP_KEEP)]
    return df.sort_values(["citizenship","year","age_mother"]).reset_index(drop=True)


def load_tfr(path: Path) -> pd.DataFrame:
    """
    Load observed TFR series. TFR carries the fertility level, while ASFR schedules carry
    the age pattern.
    """
    df = _load_csv(path)
    df = _rename_with_aliases(df, {
        "year": ["year","anno","time_period"],
        "citizenship": ["citizenship","cittadinanza","citizenship_group"],
        "tfr": ["tfr","total_fertility_rate","numero_medio_figli_per_donna","value","osservazione"],
    })
    need = ["year","citizenship","tfr"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"TFR file is missing columns: {miss}")
    df = df[need].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["tfr"] = pd.to_numeric(df["tfr"], errors="coerce")
    df["citizenship"] = df["citizenship"].astype(str).str.strip()
    df = df.dropna(subset=["year","citizenship","tfr"])
    if CITIZENSHIP_KEEP is not None:
        df = df[df["citizenship"].isin(CITIZENSHIP_KEEP)]
    return df.sort_values(["citizenship","year"]).reset_index(drop=True)


def beta_from_alpha_p_h(alpha: float, p: float, h: float) -> float:
    """
    Compute beta, the upper end of the fertility schedule.

    In Schmertmann's model, alpha marks the start of fertility, P is the peak,
    H is the age after the peak where fertility has fallen to half of its maximum,
    and beta closes the reproductive schedule. Beta is not forecast directly; it is
    derived from P and H so the curve has a plausible right tail.
    """
    # A very short right tail would make fertility fall to zero too quickly after H.
    lower = h + (h - p) / 3.0

    # A very long right tail would keep non-zero fertility at implausibly high ages.
    upper = h + 3.0 * (h - p)

    # The value 50 is used as a demographic anchor for the end of the reproductive span,
    # but it is kept inside the lower/upper bounds implied by P and H.
    return float(min(max(50.0, lower), upper))


def knots_from_alpha_p_h(alpha: float, p: float, h: float) -> np.ndarray:
    """
    Build the five knots of the Schmertmann quadratic spline.

    The knots are the ages where the spline is allowed to change curvature. They are
    placed around the start of fertility, the peak, the half-decline point, and the
    end of the schedule. This is how the intuitive parameters alpha, P, and H are
    translated into a smooth age pattern.
    """
    # When the distance between alpha and P is larger, the first internal knot is
    # allowed to move closer to the peak, giving the rising part more flexibility.
    w = min(0.75, 0.25 + 0.025 * (p - alpha))

    # Beta is derived from P and H and determines the far-right end of the schedule.
    beta = beta_from_alpha_p_h(alpha, p, h)

    # The five knots define the truncated quadratic basis used below.
    return np.array([
        alpha,
        (1.0 - w) * alpha + w * p,
        p,
        (p + h) / 2.0,
        (h + beta) / 2.0,
    ], dtype=float)


def tq_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """
    Build the truncated quadratic basis used by the Schmertmann spline.

    Each basis term is zero before its knot and grows quadratically after that knot.
    A weighted sum of these terms gives the standardized fertility curve.
    """
    x = np.asarray(x, dtype=float)
    return np.column_stack([np.maximum(0.0, x - tk) ** 2 for tk in knots])


def tq_basis_derivative(x: float, knots: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the truncated quadratic basis at one age.

    The derivative is needed to impose zero slope at the peak P and at beta.
    This makes the curve smooth at the most important points.
    """
    return np.array([2.0 * max(0.0, x - tk) for tk in knots], dtype=float)


def solve_gamma(alpha: float, p: float, h: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Solve the spline coefficients for a given alpha, P, and H.

    The five equations impose the shape constraints that make Schmertmann's model
    interpretable:
      - the curve equals 1 at the peak P;
      - the curve equals 0.5 at H;
      - the curve equals 0 at beta;
      - the slope is zero at P;
      - the slope is zero at beta.
    """
    beta = beta_from_alpha_p_h(alpha, p, h)
    knots = knots_from_alpha_p_h(alpha, p, h)

    # Matrix A contains the spline basis and derivative values at the constraint ages.
    A = np.vstack([
        tq_basis(np.array([p]), knots)[0],
        tq_basis(np.array([h]), knots)[0],
        tq_basis(np.array([beta]), knots)[0],
        tq_basis_derivative(p, knots),
        tq_basis_derivative(beta, knots),
    ])

    # Vector b contains the target values for the five constraints listed above.
    b = np.array([1.0, 0.5, 0.0, 0.0, 0.0], dtype=float)

    # Solving this linear system gives the spline weights that satisfy the constraints.
    gamma = np.linalg.solve(A, b)
    return gamma, knots, beta


def schmertmann_shape(ages: np.ndarray, alpha: float, p: float, h: float) -> np.ndarray:
    """
    Generate the standardized fertility age pattern for alpha, P, and H.

    The result is a shape, not a full ASFR schedule. It is normalized to peak at 1.
    Later, the shape is rescaled so that the sum across ages equals the forecasted TFR.
    """
    gamma, knots, beta = solve_gamma(alpha, p, h)

    # Evaluate the spline at the required ages.
    phi = tq_basis(np.asarray(ages, dtype=float), knots) @ gamma

    # Fertility is forced to zero before alpha and after beta.
    phi = np.where((ages < alpha) | (ages > beta), 0.0, phi)

    # Small negative values can appear from numerical approximation; they have no
    # demographic meaning, so we set them to zero.
    phi = np.clip(phi, 0.0, None)

    # Normalize to unit peak so this function represents timing/shape only.
    m = phi.max()
    if m > 0:
        phi = phi / m
    return phi


def observed_initial_values(ages: np.ndarray, rates: np.ndarray) -> Tuple[float,float,float]:
    """
    Build starting values for alpha, P, and H from an observed ASFR schedule.

    The optimizer needs plausible starting values. We use the observed schedule:
      - P starts at the observed peak age;
      - alpha starts at the first age with non-trivial fertility;
      - H starts at the first age after the peak where fertility falls below half
        of the peak.
    """
    rates = np.asarray(rates, dtype=float)
    ages = np.asarray(ages, dtype=float)

    # If the schedule is empty, use generic Italian fertility ages as a safe fallback.
    if np.all(rates <= 0):
        return 15.0, 30.0, 36.0

    # P is initialized at the age where observed fertility is highest.
    peak_idx = int(np.argmax(rates))
    p0 = float(ages[peak_idx])

    # Alpha is initialized as the first age where fertility is more than 1% of the peak.
    thr = 0.01 * rates.max()
    positive = ages[rates > thr]
    alpha0 = float(positive.min()) if len(positive) else max(AGE_MIN, p0 - 10)

    # H is initialized on the declining side of the curve.
    right_ages = ages[ages > p0]
    right_rates = rates[ages > p0]
    half = 0.5 * rates.max()
    if len(right_rates) == 0:
        h0 = min(p0 + 5, AGE_MAX)
    else:
        below = np.where(right_rates <= half)[0]
        h0 = float(right_ages[below[0]]) if len(below) else min(p0 + 6.0, AGE_MAX)

    # These bounds keep the starting values in the correct demographic order:
    # alpha must be before P, and H must be after P.
    alpha0 = max(AGE_MIN, min(alpha0, p0 - 3.0))
    p0 = max(alpha0 + 3.0, min(p0, AGE_MAX - 3.0))
    h0 = max(p0 + 1.0, min(h0, AGE_MAX))
    return alpha0, p0, h0


def fit_schmertmann_to_schedule(ages: np.ndarray, asfr: np.ndarray) -> Dict[str,float]:
    """
    Fit alpha, P, and H to one observed ASFR schedule.

    The ASFR schedule is normalized by its maximum before fitting. This is because
    this function estimates the age pattern of fertility, while the overall level is
    handled separately by the TFR forecast.
    """
    ages = np.asarray(ages, dtype=float)
    asfr = np.asarray(asfr, dtype=float)

    # Keep only valid non-negative fertility observations.
    mask = np.isfinite(ages) & np.isfinite(asfr) & (asfr >= 0)
    ages = ages[mask]
    asfr = asfr[mask]

    # A minimum number of ages with positive information is needed to identify a shape.
    if len(ages) < 8 or asfr.max() <= 0:
        raise ValueError("Too little information to fit Schmertmann parameters.")

    # Normalize the schedule so the optimization focuses on shape, not level.
    y = asfr / asfr.max()
    alpha0, p0, h0 = observed_initial_values(ages, y)

    def unpack(theta: np.ndarray):
        """
        Convert optimizer parameters into demographically ordered parameters.

        The exponential terms guarantee P > alpha and H > P during optimization.
        This avoids invalid schedules without needing many hard constraints.
        """
        alpha = theta[0]
        p = alpha + np.exp(theta[1])
        h = p + np.exp(theta[2])
        return alpha, p, h

    # The optimizer works on transformed parameters, so the initial values are transformed too.
    theta0 = np.array([alpha0, np.log(max(1e-3, p0-alpha0)), np.log(max(1e-3, h0-p0))], dtype=float)

    def objective(theta: np.ndarray) -> float:
        """
        Measure the distance between observed and fitted fertility shapes.

        Invalid parameter combinations are assigned a large penalty. This is simpler
        and more stable than allowing the optimizer to evaluate impossible schedules.
        """
        alpha, p, h = unpack(theta)

        if not (AGE_MIN <= alpha <= AGE_MAX - 5):
            return 1e6
        if not (alpha + 1 <= p <= AGE_MAX - 1):
            return 1e6
        if not (p + 0.5 <= h <= AGE_MAX + 5):
            return 1e6

        try:
            phi = schmertmann_shape(ages, alpha, p, h)
        except Exception:
            return 1e6

        # Higher-fertility ages get slightly more weight because they define the main
        # visible shape of the ASFR schedule.
        w = 1.0 + y
        return float(np.sum(w * (y - phi) ** 2))

    # L-BFGS-B is used because it handles simple bounds well and is stable for this
    # small numerical optimization problem.
    res = minimize(
        objective, theta0, method="L-BFGS-B",
        bounds=[(AGE_MIN, AGE_MAX-5),(np.log(0.5), np.log(30)),(np.log(0.5), np.log(20))],
        options={"maxiter": 2000}
    )

    alpha_hat, p_hat, h_hat = unpack(res.x)
    phi_hat = schmertmann_shape(ages, alpha_hat, p_hat, h_hat)
    rmse = float(np.sqrt(np.mean((y - phi_hat) ** 2)))

    # Return the fitted shape parameters and diagnostics used later in checks/plots.
    return {
        "alpha": float(alpha_hat),
        "P": float(p_hat),
        "H": float(h_hat),
        "rmse_shape": rmse,
        "peak_age_observed": float(ages[np.argmax(y)]),
    }

def fit_parameter_history(asfr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit Schmertmann parameters for every citizenship-year schedule. This converts observed
    ASFR curves into time series of alpha, P, and H.
    """
    rows = []
    for (cit, year), g in asfr_df.groupby(["citizenship","year"], dropna=False):
        g = g.sort_values("age_mother")
        try:
            est = fit_schmertmann_to_schedule(g["age_mother"].to_numpy(), g["asfr"].to_numpy())
            rows.append({"citizenship": cit, "year": int(year), **est})
        except Exception as e:
            rows.append({"citizenship": cit, "year": int(year), "alpha": np.nan, "P": np.nan, "H": np.nan,
                         "rmse_shape": np.nan, "peak_age_observed": np.nan, "fit_error": str(e)})
    return pd.DataFrame(rows).sort_values(["citizenship","year"]).reset_index(drop=True)


@dataclass
class ForecastResult:
    mean: pd.DataFrame
    simulations: pd.DataFrame
    model_summary: pd.DataFrame


def choose_arima(series: pd.Series, candidates: list[tuple[int, int, int]], fixed_order=None):
    """
    Select a time-series model for a demographic parameter by comparing candidate ARIMA
    specifications. The selected model is then used for forecast and uncertainty.
    """
    y = series.astype(float).dropna()

    if fixed_order is not None:
        p, d, q = fixed_order
        # Use intercept for stationary ARIMA, drift for differenced ARIMA
        trend = "c" if d == 0 else "t"
        model = ARIMA(y, order=fixed_order, trend=trend).fit()
        return fixed_order, model

    if len(y) < 8:
        order = (0, 1, 0)
        model = ARIMA(y, order=order, trend="t").fit()
        return order, model

    best_aic = np.inf
    best_order = None
    best_model = None

    for order in candidates:
        try:
            p, d, q = order
            trend = "c" if d == 0 else "t"
            res = ARIMA(y, order=order, trend=trend).fit()
            if np.isfinite(res.aic) and res.aic < best_aic:
                best_aic = res.aic
                best_order = order
                best_model = res
        except Exception:
            continue

    if best_model is None:
        best_order = (0, 1, 0)
        best_model = ARIMA(y, order=best_order, trend="t").fit()

    return best_order, best_model


def simulate_arima(model, steps: int, n_sim: int, seed: int = 123) -> np.ndarray:
    """
    Simulate future paths from a fitted ARIMA model. These simulated paths are what later
    become prediction intervals.
    """
    rs = np.random.RandomState(seed)
    try:
        sim = model.simulate(nsimulations=steps, repetitions=n_sim, anchor="end")
        sim = np.asarray(sim)
        if sim.shape == (steps, n_sim):
            sim = sim.T
        elif sim.shape != (n_sim, steps):
            sim = np.reshape(sim, (n_sim, steps))
        return sim
    except Exception:
        fc = model.get_forecast(steps=steps)
        mu = np.asarray(fc.predicted_mean, dtype=float)
        try:
            se = np.asarray(fc.se_mean, dtype=float)
        except Exception:
            resid_sd = float(np.nanstd(model.resid))
            se = np.repeat(resid_sd, steps)
        resid_sd = float(np.nanstd(model.resid)) if np.isfinite(np.nanstd(model.resid)) else 0.05
        se = np.where(np.isfinite(se) & (se > 0), se, resid_sd)
        return rs.normal(loc=mu, scale=se, size=(n_sim, steps))


def forecast_one_series(
    series_df: pd.DataFrame,
    value_col: str,
    horizon_years: list[int],
    citizenship: str,
    n_sim: int,
    fixed_order=None,
    candidates=None,
) -> ForecastResult:
    y = series_df.sort_values("year").set_index("year")[value_col].astype(float)

    if candidates is None:
        candidates = ARIMA_CANDIDATES

    order, model = choose_arima(y, candidates=candidates, fixed_order=fixed_order)
    steps = len(horizon_years)

    fc = model.get_forecast(steps=steps)
    mean_df = pd.DataFrame(
        {
            "citizenship": citizenship,
            "year": horizon_years,
            value_col: np.asarray(fc.predicted_mean, dtype=float),
        }
    )

    sims = simulate_arima(model, steps=steps, n_sim=n_sim, seed=RANDOM_SEED)
    sim_df = (
        pd.DataFrame(sims, columns=horizon_years)
        .assign(sim=np.arange(1, n_sim + 1), citizenship=citizenship)
        .melt(id_vars=["sim", "citizenship"], var_name="year", value_name=value_col)
        .sort_values(["sim", "year"])
        .reset_index(drop=True)
    )

    summary = pd.DataFrame(
        {
            "citizenship": [citizenship],
            "value": [value_col],
            "arima_order": [str(order)],
            "aic": [float(model.aic)],
            "n_obs": [int(y.notna().sum())],
        }
    )
    return ForecastResult(mean=mean_df, simulations=sim_df, model_summary=summary)



def forecast_foreign_tfr_from_gap(
    tfr_df: pd.DataFrame,
    italian_mean_df: pd.DataFrame,
    italian_sim_df: pd.DataFrame,
    horizon_years: list[int],
    n_sim: int,
):
    """
    Forecast foreign TFR as:
        foreign_tfr = italian_tfr + gap
    where gap is forecast on a soft, positive scale.

    Steps:
    1. build historical gap = foreign - italian
    2. remove minimum gap and forecast log(excess_gap)
    3. build a soft target path that converges gradually
    4. blend ARIMA gap forecast with the soft target path
    """

    # -------------------------
    # Historical gap
    # -------------------------
    tfr_hist = (
        tfr_df.pivot(index="year", columns="citizenship", values="tfr")
        .reset_index()
        .sort_values("year")
    )

    needed = {ITALIAN_LABEL, FOREIGN_LABEL}
    if not needed.issubset(set(tfr_hist.columns)):
        raise ValueError(f"TFR history must contain {needed}, found {set(tfr_hist.columns)}")

    tfr_hist["gap"] = tfr_hist[FOREIGN_LABEL] - tfr_hist[ITALIAN_LABEL]
    tfr_hist["gap"] = np.maximum(tfr_hist["gap"], FOREIGN_MIN_GAP + EPS)

    # Excess gap above the minimum gap
    tfr_hist["gap_excess"] = np.maximum(tfr_hist["gap"] - FOREIGN_MIN_GAP, EPS)
    tfr_hist["log_gap_excess"] = np.log(tfr_hist["gap_excess"])

    # -------------------------
    # Forecast log(excess gap)
    # -------------------------
    gap_series = tfr_hist[["year", "log_gap_excess"]].dropna()

    fr_gap = forecast_one_series(
        series_df=gap_series,
        value_col="log_gap_excess",
        horizon_years=horizon_years,
        citizenship=FOREIGN_LABEL,
        n_sim=n_sim,
        fixed_order=FOREIGN_GAP_ORDER,
        candidates=None,
    )

    # Convert back from log scale
    gap_mean = fr_gap.mean.copy()
    gap_mean["gap_excess_arima"] = np.exp(gap_mean["log_gap_excess"])
    gap_mean = gap_mean.drop(columns="log_gap_excess")

    gap_sims = fr_gap.simulations.copy()
    gap_sims["gap_excess_arima"] = np.exp(gap_sims["log_gap_excess"])
    gap_sims = gap_sims.drop(columns="log_gap_excess")

    # -------------------------
    # Soft target path
    # -------------------------
    last_gap = float(tfr_hist["gap"].iloc[-1])

    n_years = len(horizon_years)
    blend_weights = np.linspace(FOREIGN_GAP_BLEND_START, FOREIGN_GAP_BLEND_END, n_years)

    target_gap = np.array([
        FOREIGN_MIN_GAP + (last_gap - FOREIGN_MIN_GAP) * np.exp(-FOREIGN_GAP_DECAY * i)
        for i in range(n_years)
    ])

    gap_mean["blend_w"] = blend_weights
    gap_mean["gap_target"] = target_gap

    # ARIMA gap = minimum gap + excess
    gap_mean["gap_arima"] = FOREIGN_MIN_GAP + gap_mean["gap_excess_arima"]

    # Soft blend between ARIMA and target path
    gap_mean["gap_final"] = (
        gap_mean["blend_w"] * gap_mean["gap_arima"]
        + (1 - gap_mean["blend_w"]) * gap_mean["gap_target"]
    )

    gap_mean["gap_final"] = np.maximum(gap_mean["gap_final"], FOREIGN_MIN_GAP)

    # -------------------------
    # Mean foreign TFR
    # -------------------------
    foreign_mean = italian_mean_df.merge(
        gap_mean[["year", "gap_final"]],
        on="year",
        how="left"
    ).copy()

    foreign_mean["citizenship"] = FOREIGN_LABEL
    foreign_mean["tfr"] = foreign_mean["tfr"] + foreign_mean["gap_final"]
    foreign_mean = foreign_mean[["citizenship", "year", "tfr"]]

    # -------------------------
    # Simulation foreign TFR
    # -------------------------
    year_to_weight = dict(zip(horizon_years, blend_weights))
    year_to_target = dict(zip(horizon_years, target_gap))

    gap_sims["blend_w"] = gap_sims["year"].map(year_to_weight)
    gap_sims["gap_target"] = gap_sims["year"].map(year_to_target)
    gap_sims["gap_arima"] = FOREIGN_MIN_GAP + gap_sims["gap_excess_arima"]

    gap_sims["gap_final"] = (
        gap_sims["blend_w"] * gap_sims["gap_arima"]
        + (1 - gap_sims["blend_w"]) * gap_sims["gap_target"]
    )
    gap_sims["gap_final"] = np.maximum(gap_sims["gap_final"], FOREIGN_MIN_GAP)

    foreign_sims = italian_sim_df.merge(
        gap_sims[["sim", "year", "gap_final"]],
        on=["sim", "year"],
        how="left"
    ).copy()

    foreign_sims["citizenship"] = FOREIGN_LABEL
    foreign_sims["tfr"] = foreign_sims["tfr"] + foreign_sims["gap_final"]
    foreign_sims = foreign_sims[["sim", "citizenship", "year", "tfr"]]

    # -------------------------
    # Model summary
    # -------------------------
    model_summary = fr_gap.model_summary.copy()
    model_summary["value"] = "foreign_gap_log_excess"
    model_summary["notes"] = (
        "Foreign TFR forecasted as Italian TFR plus a softly declining positive gap"
    )

    return foreign_mean, foreign_sims, model_summary

def smooth_and_constrain_schmertmann_params(
    mean_cit: pd.DataFrame,
    sim_cit: pd.DataFrame,
    ph_hist: pd.DataFrame,
):
    """
    Apply soft demographic constraints and shrink forecasted alpha, P, H
    toward recent historical means.
    """

    mean_cit = mean_cit.copy()
    sim_cit = sim_cit.copy()
    ph_hist = ph_hist.sort_values("year").copy()

    recent = ph_hist.tail(PARAM_TARGET_WINDOW)
    alpha_target = float(recent["alpha"].mean())
    p_target = float(recent["P"].mean())
    h_target = float(recent["H"].mean())

    years_sorted = sorted(mean_cit["year"].unique())
    n_years = len(years_sorted)
    blend_weights = np.linspace(PARAM_BLEND_START, PARAM_BLEND_END, n_years)
    weight_map = dict(zip(years_sorted, blend_weights))

    # ---- mean forecast
    if USE_PARAM_SHRINKAGE:
        mean_cit["blend_w"] = mean_cit["year"].map(weight_map)
        mean_cit["alpha"] = mean_cit["blend_w"] * mean_cit["alpha"] + (1 - mean_cit["blend_w"]) * alpha_target
        mean_cit["P"] = mean_cit["blend_w"] * mean_cit["P"] + (1 - mean_cit["blend_w"]) * p_target
        mean_cit["H"] = mean_cit["blend_w"] * mean_cit["H"] + (1 - mean_cit["blend_w"]) * h_target
        mean_cit = mean_cit.drop(columns="blend_w")

    mean_cit["alpha"] = mean_cit["alpha"].clip(ALPHA_MIN, ALPHA_MAX)
    mean_cit["P"] = mean_cit["P"].clip(P_MIN, P_MAX)
    mean_cit["H"] = mean_cit["H"].clip(H_MIN, H_MAX)

    mean_cit["P"] = np.maximum(mean_cit["P"], mean_cit["alpha"] + MIN_P_MINUS_ALPHA)
    mean_cit["H"] = np.maximum(mean_cit["H"], mean_cit["P"] + MIN_H_MINUS_P)

    # ---- simulations
    if USE_PARAM_SHRINKAGE:
        sim_cit["blend_w"] = sim_cit["year"].map(weight_map)
        sim_cit["alpha"] = sim_cit["blend_w"] * sim_cit["alpha"] + (1 - sim_cit["blend_w"]) * alpha_target
        sim_cit["P"] = sim_cit["blend_w"] * sim_cit["P"] + (1 - sim_cit["blend_w"]) * p_target
        sim_cit["H"] = sim_cit["blend_w"] * sim_cit["H"] + (1 - sim_cit["blend_w"]) * h_target
        sim_cit = sim_cit.drop(columns="blend_w")

    sim_cit["alpha"] = sim_cit["alpha"].clip(ALPHA_MIN, ALPHA_MAX)
    sim_cit["P"] = sim_cit["P"].clip(P_MIN, P_MAX)
    sim_cit["H"] = sim_cit["H"].clip(H_MIN, H_MAX)

    sim_cit["P"] = np.maximum(sim_cit["P"], sim_cit["alpha"] + MIN_P_MINUS_ALPHA)
    sim_cit["H"] = np.maximum(sim_cit["H"], sim_cit["P"] + MIN_H_MINUS_P)

    return mean_cit, sim_cit


def forecast_parameters_and_tfr(param_hist: pd.DataFrame, tfr_df: pd.DataFrame, start_year: int, end_year: int, n_sim: int):
    """
    Forecast all fertility inputs needed to rebuild future ASFR: TFR plus the Schmertmann
    age-shape parameters.
    """
    years = list(range(start_year, end_year + 1))
    model_info = []

    # ----------------------------------------------------
    # 1. TFR block
    # ----------------------------------------------------

    # Italians: direct ARIMA in levels
    tfr_it = tfr_df.loc[tfr_df["citizenship"] == ITALIAN_LABEL, ["year", "tfr"]].dropna()
    if tfr_it.empty:
        raise ValueError(f"No TFR data found for {ITALIAN_LABEL}")

    fr_it = forecast_one_series(
        series_df=tfr_it,
        value_col="tfr",
        horizon_years=years,
        citizenship=ITALIAN_LABEL,
        n_sim=n_sim,
        fixed_order=ITALIAN_TFR_ORDER,
        candidates=None,
    )
    model_info.append(fr_it.model_summary)

    italian_mean_tfr = fr_it.mean.copy()
    italian_sim_tfr = fr_it.simulations.copy()

    # Foreigners: Italians + soft positive gap
    foreign_mean_tfr, foreign_sim_tfr, foreign_model_summary = forecast_foreign_tfr_from_gap(
        tfr_df=tfr_df,
        italian_mean_df=italian_mean_tfr,
        italian_sim_df=italian_sim_tfr,
        horizon_years=years,
        n_sim=n_sim,
    )
    model_info.append(foreign_model_summary)

    tfr_mean_df = pd.concat([italian_mean_tfr, foreign_mean_tfr], ignore_index=True)
    tfr_sim_df = pd.concat([italian_sim_tfr, foreign_sim_tfr], ignore_index=True)

    # ----------------------------------------------------
    # 2. Schmertmann parameter block
    # ----------------------------------------------------
    common_cit = sorted(set(param_hist["citizenship"]).intersection(set(tfr_df["citizenship"])))
    mean_param_all = []
    sim_param_all = []

    for cit in common_cit:
        ph = param_hist.loc[param_hist["citizenship"] == cit, ["year", "alpha", "P", "H"]].dropna()

        if ph.empty:
            raise ValueError(
                f"No valid Schmertmann parameter history for citizenship '{cit}'. "
                f"Check schmertmann_parameter_history.csv."
            )

        mean_cit = tfr_mean_df[tfr_mean_df["citizenship"] == cit].copy()
        sim_cit = tfr_sim_df[tfr_sim_df["citizenship"] == cit].copy()

        for col in ["alpha", "P", "H"]:
            fr_par = forecast_one_series(
                series_df=ph[["year", col]].dropna(),
                value_col=col,
                horizon_years=years,
                citizenship=cit,
                n_sim=n_sim,
                fixed_order=None,
                candidates=ARIMA_CANDIDATES_PARAMS,
            )
            model_info.append(fr_par.model_summary)

            mean_cit = mean_cit.merge(fr_par.mean, on=["citizenship", "year"], how="left")
            sim_cit = sim_cit.merge(fr_par.simulations, on=["sim", "citizenship", "year"], how="left")

        # NEW: soft constraints + shrinkage
        mean_cit, sim_cit = smooth_and_constrain_schmertmann_params(
            mean_cit=mean_cit,
            sim_cit=sim_cit,
            ph_hist=ph
        )

        mean_param_all.append(mean_cit)
        sim_param_all.append(sim_cit)

    mean_df = pd.concat(mean_param_all, ignore_index=True).sort_values(["citizenship", "year"]).reset_index(drop=True)
    sim_df = pd.concat(sim_param_all, ignore_index=True).sort_values(["sim", "citizenship", "year"]).reset_index(drop=True)

    # ----------------------------------------------------
    # 3. Final guards
    # ----------------------------------------------------
    def enforce(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        required = ["tfr", "alpha", "P", "H"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing forecast columns after merge: {missing}")

        df["alpha"] = df["alpha"].clip(ALPHA_MIN, ALPHA_MAX)
        df["P"] = df["P"].clip(P_MIN, P_MAX)
        df["H"] = df["H"].clip(H_MIN, H_MAX)

        df["P"] = np.maximum(df["P"], df["alpha"] + MIN_P_MINUS_ALPHA)
        df["H"] = np.maximum(df["H"], df["P"] + MIN_H_MINUS_P)

        df["tfr"] = df["tfr"].clip(lower=0.0)
        return df

    model_df = pd.concat(model_info, ignore_index=True)
    return enforce(mean_df), enforce(sim_df), model_df

# def forecast_parameters_and_tfr(param_hist: pd.DataFrame, tfr_df: pd.DataFrame, start_year: int, end_year: int, n_sim: int):
#     years = list(range(start_year, end_year + 1))
#     mean_all = []
#     sim_all = []
#     model_info = []

#     # ----------------------------------------------------
#     # 1. TFR block
#     # ----------------------------------------------------

#     # ---- Italians first: direct ARIMA in levels
#     tfr_it = tfr_df.loc[tfr_df["citizenship"] == ITALIAN_LABEL, ["year", "tfr"]].dropna()
#     if tfr_it.empty:
#         raise ValueError(f"No TFR data found for {ITALIAN_LABEL}")

#     fr_it = forecast_one_series(
#         series_df=tfr_it,
#         value_col="tfr",
#         horizon_years=years,
#         citizenship=ITALIAN_LABEL,
#         n_sim=n_sim,
#         fixed_order=ITALIAN_TFR_ORDER,
#         candidates=None,
#     )
#     model_info.append(fr_it.model_summary)

#     italian_mean_tfr = fr_it.mean.copy()
#     italian_sim_tfr = fr_it.simulations.copy()

#     # ---- Foreigners: forecast the gap over Italians
#     foreign_mean_tfr, foreign_sim_tfr, foreign_model_summary = forecast_foreign_tfr_from_gap(
#         tfr_df=tfr_df,
#         italian_mean_df=italian_mean_tfr,
#         italian_sim_df=italian_sim_tfr,
#         horizon_years=years,
#         n_sim=n_sim,
#     )
#     model_info.append(foreign_model_summary)

#     tfr_mean_df = pd.concat([italian_mean_tfr, foreign_mean_tfr], ignore_index=True)
#     tfr_sim_df = pd.concat([italian_sim_tfr, foreign_sim_tfr], ignore_index=True)

#     # ----------------------------------------------------
#     # 2. Schmertmann parameter block: alpha, P, H
#     # ----------------------------------------------------
#     common_cit = sorted(set(param_hist["citizenship"]).intersection(set(tfr_df["citizenship"])))

#     mean_param_all = []
#     sim_param_all = []

#     for cit in common_cit:
#         ph = param_hist.loc[param_hist["citizenship"] == cit, ["year", "alpha", "P", "H"]].dropna()

#         if ph.empty:
#             raise ValueError(
#                 f"No valid Schmertmann parameter history for citizenship '{cit}'. "
#                 f"Check the fit step and inspect schmertmann_parameter_history.csv."
#             )

#         # start from TFR block
#         mean_cit = tfr_mean_df[tfr_mean_df["citizenship"] == cit].copy()
#         sim_cit = tfr_sim_df[tfr_sim_df["citizenship"] == cit].copy()

#         for col in ["alpha", "P", "H"]:
#             fr_par = forecast_one_series(
#                 series_df=ph[["year", col]].dropna(),
#                 value_col=col,
#                 horizon_years=years,
#                 citizenship=cit,
#                 n_sim=n_sim,
#                 fixed_order=None,
#                 candidates=ARIMA_CANDIDATES_PARAMS,
#             )
#             model_info.append(fr_par.model_summary)

#             mean_cit = mean_cit.merge(fr_par.mean, on=["citizenship", "year"], how="left")
#             sim_cit = sim_cit.merge(fr_par.simulations, on=["sim", "citizenship", "year"], how="left")

#         mean_param_all.append(mean_cit)
#         sim_param_all.append(sim_cit)

#     mean_df = pd.concat(mean_param_all, ignore_index=True).sort_values(["citizenship", "year"]).reset_index(drop=True)
#     sim_df = pd.concat(sim_param_all, ignore_index=True).sort_values(["sim", "citizenship", "year"]).reset_index(drop=True)

#     # ----------------------------------------------------
#     # 3. Final parameter guards
#     # ----------------------------------------------------
#     def enforce(df: pd.DataFrame) -> pd.DataFrame:
#         df = df.copy()

#         required = ["tfr", "alpha", "P", "H"]
#         missing = [c for c in required if c not in df.columns]
#         if missing:
#             raise ValueError(f"Missing forecast columns after merge: {missing}")

#         df["alpha"] = df["alpha"].clip(lower=AGE_MIN, upper=AGE_MAX - 5)
#         df["P"] = np.maximum(df["P"], df["alpha"] + 1.0)
#         df["P"] = np.minimum(df["P"], AGE_MAX - 1.0)
#         df["H"] = np.maximum(df["H"], df["P"] + 0.5)
#         df["H"] = np.minimum(df["H"], AGE_MAX + 5.0)
#         df["tfr"] = df["tfr"].clip(lower=0.0)

#         return df

#     model_df = pd.concat(model_info, ignore_index=True)
#     return enforce(mean_df), enforce(sim_df), model_df


def asfr_from_tfr_and_shape(ages: np.ndarray, tfr: float, alpha: float, p: float, h: float) -> np.ndarray:
    """
    Combine a forecasted TFR with a Schmertmann shape. The shape gives the age distribution,
    and rescaling makes the ASFR sum to the desired TFR.
    """
    phi = schmertmann_shape(ages, alpha, p, h)
    s = phi.sum()
    if s <= 0 or not np.isfinite(s):
        return np.zeros_like(ages, dtype=float)
    return (tfr / s) * phi


def build_future_schedule(mean_df: pd.DataFrame, sim_df: pd.DataFrame, ages: np.ndarray):
    """
    Construct future ASFR schedules for the mean forecast and for the simulation paths.
    """
    mean_rows = []
    for _, row in mean_df.iterrows():
        asfr = asfr_from_tfr_and_shape(ages, row["tfr"], row["alpha"], row["P"], row["H"])
        mean_rows.append(pd.DataFrame({
            "citizenship": row["citizenship"], "year": row["year"], "age_mother": ages, "asfr": asfr,
            "tfr": row["tfr"], "alpha": row["alpha"], "P": row["P"], "H": row["H"]
        }))
    mean_schedule = pd.concat(mean_rows, ignore_index=True)

    sim_rows = []
    for (sim, cit, year), rowg in sim_df.groupby(["sim","citizenship","year"]):
        row = rowg.iloc[0]
        asfr = asfr_from_tfr_and_shape(ages, row["tfr"], row["alpha"], row["P"], row["H"])
        sim_rows.append(pd.DataFrame({
            "sim": sim, "citizenship": cit, "year": year, "age_mother": ages, "asfr": asfr,
            "tfr": row["tfr"], "alpha": row["alpha"], "P": row["P"], "H": row["H"]
        }))
    sim_schedule = pd.concat(sim_rows, ignore_index=True)
    return mean_schedule, sim_schedule


def merge_history_and_forecast_asfr(asfr_df: pd.DataFrame, future_schedule_mean: pd.DataFrame) -> pd.DataFrame:
    """
    Join observed ASFR and forecasted ASFR in one file so figures and projection inputs can
    show a continuous historical-forecast series.
    """
    hist = asfr_df[["citizenship","year","age_mother","asfr"]].copy()
    hist["source"] = "observed"
    fut = future_schedule_mean[["citizenship","year","age_mother","asfr"]].copy()
    fut["source"] = "forecast"
    return pd.concat([hist,fut], ignore_index=True).sort_values(["citizenship","year","age_mother"]).reset_index(drop=True)


def main():
    """
    Run the complete script from inputs to outputs.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    asfr_df = load_asfr(ASFR_FILE)
    asfr_df["asfr"] = pd.to_numeric(asfr_df["asfr"], errors="coerce")
    asfr_df["asfr"] = asfr_df["asfr"] / 1000
    tfr_df = load_tfr(TFR_FILE)
    common = sorted(set(asfr_df["citizenship"]).intersection(set(tfr_df["citizenship"])))
    asfr_df = asfr_df[asfr_df["citizenship"].isin(common)].copy()
    tfr_df = tfr_df[tfr_df["citizenship"].isin(common)].copy()
    param_hist = fit_parameter_history(asfr_df)
    param_hist.to_csv(OUTPUT_DIR / "schmertmann_parameter_history.csv", index=False)
    last_year = int(min(asfr_df["year"].max(), tfr_df["year"].max()))
    start_year = int(last_year + 1) if FORECAST_START is None else int(FORECAST_START)
    mean_df, sim_df, model_df = forecast_parameters_and_tfr(param_hist, tfr_df, start_year, FORECAST_END, N_SIM)
    mean_df.to_csv(OUTPUT_DIR / "schmertmann_parameter_forecast_mean.csv", index=False)
    sim_df.to_csv(OUTPUT_DIR / "schmertmann_parameter_forecast_simulations.csv", index=False)
    model_df.to_csv(OUTPUT_DIR / "forecast_model_choices.csv", index=False)
    mean_df[["citizenship","year","tfr"]].to_csv(OUTPUT_DIR / "tfr_forecast_mean.csv", index=False)
    sim_df[["sim","citizenship","year","tfr"]].to_csv(OUTPUT_DIR / "tfr_forecast_simulations.csv", index=False)
    tfr_mean = mean_df[["citizenship", "year", "tfr"]].copy()
    tfr_sims = sim_df[["sim", "citizenship", "year", "tfr"]].copy()

    tfr_summary = (
        tfr_sims.groupby(["citizenship", "year"])["tfr"]
        .quantile([0.10, 0.50, 0.90])
        .unstack()
        .reset_index()
        .rename(columns={0.10: "tfr_p10", 0.50: "tfr_p50", 0.90: "tfr_p90"})
        .merge(tfr_mean.rename(columns={"tfr": "tfr_mean"}), on=["citizenship", "year"], how="left")
    )

    order_lookup = model_df[model_df["value"].isin(["tfr", "foreign_gap_log_excess"])][["citizenship", "arima_order"]].copy()
    order_lookup = order_lookup.groupby("citizenship", as_index=False)["arima_order"].first()

    tfr_summary = tfr_summary.merge(order_lookup, on="citizenship", how="left")
    tfr_summary = tfr_summary[["citizenship", "year", "tfr_mean", "tfr_p10", "tfr_p50", "tfr_p90", "arima_order"]]
    tfr_summary.to_csv(OUTPUT_DIR / "tfr_forecast_summary.csv", index=False)
    ages = np.arange(AGE_MIN, AGE_MAX + 1)
    mean_schedule, sim_schedule = build_future_schedule(mean_df, sim_df, ages)
    mean_schedule.to_csv(OUTPUT_DIR / "fertility_forecast_schedule_mean.csv", index=False)
    sim_schedule.to_csv(OUTPUT_DIR / "fertility_forecast_schedule_simulations.csv", index=False)
    merge_history_and_forecast_asfr(asfr_df, mean_schedule).to_csv(OUTPUT_DIR / "fertility_history_plus_forecast_mean.csv", index=False)
    print(f"Done. Output written to: {OUTPUT_DIR.resolve()}")
    for p in sorted(OUTPUT_DIR.glob("*.csv")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
