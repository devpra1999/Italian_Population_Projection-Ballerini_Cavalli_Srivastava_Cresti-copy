# =============================================================================
# MORTALITY MODULE - LEE-MILLER ESTIMATION AND FORECAST
# =============================================================================
# Reading guide
# -------------
# This script estimates and forecasts age-specific mortality probabilities qx.
# The core model is Lee-Carter in a Lee-Miller style:
#   log(qx,t) = ax + bx * kt.
#
# The demographic idea is:
#   - ax captures the average age pattern of mortality;
#   - bx captures which ages respond more to changes in overall mortality;
#   - kt captures the period trend in mortality over time.
#
# The script estimates ax, bx, and kt separately by sex, forecasts kt, simulates
# uncertainty, reconstructs future qx schedules, and derives life expectancy at
# birth. ISTAT/UN/Eurostat benchmarks are used only for comparison.
# =============================================================================

"""
02_fit_forecast_mortality_leemiller_v2.py

Mortality module for the projection project.

What it does
------------
1. Reads historical mortality schedules by year x age x sex.
2. Fits Lee-Carter / Lee-Miller mortality models by sex.
3. Runs two model versions:
   - all_years: uses all observed years;
   - excluding_covid: excludes COVID shock years from the model fit.
4. Forecasts the Lee-Miller time index k_t with a random walk with drift.
5. Simulates future k_t paths and reconstructs future qx, mx, and px schedules.
6. Calculates life expectancy at birth (e0) with uncertainty.
7. Exports forecasted qx age-by-age and year-by-year, plus e0 summaries.

Expected input
--------------
data/processed/mortality.csv  or  mortality.csv

Required columns, with flexible aliases accepted:
    year, age, sex, qx  OR  year, age, sex, mx

Optional benchmark input
------------------------
data/processed/mortality_benchmarks_e0.csv

Columns:
    source, year, sex, e0
Optional:
    e0_lower, e0_upper, scenario

Main outputs
------------
output/mortality/all_years/
output/mortality/excluding_covid/
output/mortality/comparison/

Each model folder contains:
    lee_miller_age_parameters.csv
    lee_miller_time_index_history.csv
    lee_miller_time_index_forecast_mean.csv
    lee_miller_time_index_forecast_simulations.csv
    mortality_fit_observed_vs_fitted.csv
    mortality_forecast_schedule_mean.csv
    mortality_forecast_schedule_simulations.csv
    survival_forecast_mean.csv
    survival_forecast_simulations.csv
    life_expectancy_history.csv
    life_expectancy_forecast_mean.csv
    life_expectancy_forecast_simulations.csv
    life_expectancy_forecast_intervals.csv
    mortality_history_plus_forecast_mean.csv
    forecast_model_choices.csv

Notes
-----
- qx is preferred. If only mx is available, qx is approximated as 1 - exp(-mx).
- The COVID exclusion is controlled through COVID_YEARS below. Default: 2020, 2021.
- Forecasts always start after the last observed year in the full input, not after the
  last year used in the fit. This keeps all_years and excluding_covid directly comparable.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

PROJECT_DIR = Path("/Users/andreaballerini/Downloads/projection_project")
INPUT_DIR = PROJECT_DIR
OUTPUT_ROOT = PROJECT_DIR / "output" / "mortality"

MORTALITY_FILE_CANDIDATES = [
    INPUT_DIR / "mortality.csv",
    INPUT_DIR / "data" / "processed" / "mortality.csv",
    INPUT_DIR / "qx.csv",
]

WPP_QX_FILE_CANDIDATES = {
    "female": [
        INPUT_DIR / "WPP Female qx.xlsx",
        INPUT_DIR / "data" / "processed" / "WPP Female qx.xlsx",
        INPUT_DIR / "wpp_female_qx.xlsx",
    ],
    "male": [
        INPUT_DIR / "WPP Male qx.xlsx",
        INPUT_DIR / "data" / "processed" / "WPP Male qx.xlsx",
        INPUT_DIR / "wpp_male_qx.xlsx",
    ],
}

E0_OTHER_FILE_CANDIDATES = [
    INPUT_DIR / "e0altri.xlsx",
    INPUT_DIR / "data" / "processed" / "e0altri.xlsx",
    INPUT_DIR / "e0_other.xlsx",
]

BENCHMARK_FILE_CANDIDATES = [
    INPUT_DIR / "data" / "processed" / "mortality_benchmarks_e0.csv",
    INPUT_DIR / "mortality_benchmarks_e0.csv",
    INPUT_DIR / "e0_benchmarks.csv",
]

FORECAST_START: Optional[int] = None
FORECAST_END = 2075
N_SIM = 3000
MAX_AGE = 100
RANDOM_SEED = 123
COVID_YEARS = [2020, 2021]
ARIMA_ORDER_KT = (0, 1, 0)  # random walk with drift via trend='t'

EPS = 1e-12
QX_FLOOR = 1e-9
QX_CAP = 0.999999

np.random.seed(RANDOM_SEED)

MODEL_RUNS = {
    "all_years": [],
    "excluding_covid": COVID_YEARS,
}


# ============================================================
# I/O helpers
# ============================================================
def _normalize_colname(s: str) -> str:
    """
    Normalize column names so that small differences in accents, spaces, and capitalization
    do not break the import step.
    """
    s = str(s).strip().lower()
    for a, b in [("à", "a"), ("é", "e"), ("è", "e"), ("ì", "i"), ("ò", "o"), ("ù", "u")]:
        s = s.replace(a, b)
    for x in ["(", ")", "/", "-", " ", "."]:
        s = s.replace(x, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


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
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    return pd.read_csv(path)


def _find_input_file(candidates: List[Path]) -> Path:
    """
    Search the project folder for the first available input among accepted alternatives.
    This keeps the script runnable when filenames change slightly.
    """
    for path in candidates:
        if path.exists():
            return path
    tried = "\n - ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"No mortality input file found. Tried:\n - {tried}")


def _find_optional_file(candidates: List[Path]) -> Optional[Path]:
    """
    Find an optional benchmark file if it exists, without stopping the model when that
    comparison source is absent.
    """
    for path in candidates:
        if path.exists():
            return path
    return None


# ============================================================
# Mortality utilities
# ============================================================
def qx_from_mx(mx: np.ndarray) -> np.ndarray:
    """
    Convert central death rates mx into death probabilities qx. The projection engine works
    more naturally with qx schedules.
    """
    mx = np.asarray(mx, dtype=float)
    qx = 1.0 - np.exp(-np.clip(mx, 0.0, None))
    return np.clip(qx, QX_FLOOR, QX_CAP)


def mx_from_qx(qx: np.ndarray) -> np.ndarray:
    """
    Convert death probabilities qx back into central rates mx when needed for comparison or
    diagnostics.
    """
    qx = np.asarray(qx, dtype=float)
    qx = np.clip(qx, QX_FLOOR, QX_CAP)
    return -np.log(1.0 - qx)


def qx_to_life_table(qx: np.ndarray, ages: Optional[np.ndarray] = None, radix: float = 100000.0) -> pd.DataFrame:
    """
    Build a simple life table from age-specific death probabilities. This is used to derive
    life expectancy from each forecasted mortality schedule.
    """
    qx = np.asarray(qx, dtype=float)
    qx = np.clip(qx, QX_FLOOR, QX_CAP)
    n = len(qx)
    if ages is None:
        ages = np.arange(n)
    else:
        ages = np.asarray(ages, dtype=int)

    lx = np.zeros(n + 1, dtype=float)
    dx = np.zeros(n, dtype=float)
    Lx = np.zeros(n, dtype=float)
    Tx = np.zeros(n, dtype=float)
    ex = np.zeros(n, dtype=float)

    lx[0] = radix
    for i in range(n):
        dx[i] = lx[i] * qx[i]
        lx[i + 1] = max(0.0, lx[i] - dx[i])

    for i in range(n - 1):
        Lx[i] = 0.5 * (lx[i] + lx[i + 1])

    q_last = min(max(qx[-1], QX_FLOOR), QX_CAP)
    m_last = -np.log(1.0 - q_last)
    Lx[-1] = lx[-1] / max(m_last, EPS)

    running = 0.0
    for i in range(n - 1, -1, -1):
        running += Lx[i]
        Tx[i] = running
        ex[i] = Tx[i] / max(lx[i], EPS)

    return pd.DataFrame({"age": ages, "qx": qx, "lx": lx[:-1], "dx": dx, "Lx": Lx, "Tx": Tx, "ex": ex})


def e0_from_qx(qx: np.ndarray) -> float:
    """
    Return life expectancy at birth from a qx schedule. This provides a compact summary of
    the full mortality curve.
    """
    return float(qx_to_life_table(qx).loc[0, "ex"])


def reconstruct_qx(ax: np.ndarray, bx: np.ndarray, kt: float) -> np.ndarray:
    """
    Reconstruct age-specific mortality from Lee-Carter/Lee-Miller parameters ax, bx, and kt.
    """
    log_qx = ax + bx * kt
    qx = np.exp(log_qx)
    return np.clip(qx, QX_FLOOR, QX_CAP)


def lee_miller_adjust_kt(ax: np.ndarray, bx: np.ndarray, target_e0: float, kt_start: float) -> float:
    """
    Adjust kt so that the reconstructed mortality schedule matches a target life expectancy.
    This follows the Lee-Miller logic of aligning schedules with e0.
    """
    def objective(kt: float) -> float:
        return e0_from_qx(reconstruct_qx(ax, bx, kt)) - target_e0

    lo, hi = -200.0, 200.0
    f_lo, f_hi = objective(lo), objective(hi)
    if np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo * f_hi < 0:
        return float(brentq(objective, lo, hi, maxiter=200))

    res = minimize_scalar(lambda z: objective(z) ** 2, bracket=(kt_start - 20, kt_start, kt_start + 20), method="brent")
    return float(res.x)


def standardize_sex(s: pd.Series) -> pd.Series:
    """
    Standardize sex labels so male/female series are handled consistently across input and
    benchmark files.
    """
    out = s.astype(str).str.strip().str.lower()
    mapping = {
        "m": "male", "maschi": "male", "maschio": "male", "male": "male", "men": "male", "uomini": "male", "uomo": "male", "1": "male",
        "f": "female", "femmine": "female", "femmina": "female", "female": "female", "women": "female", "donne": "female", "donna": "female", "2": "female",
        "totale": "total", "total": "total", "t": "total", "both": "total",
    }
    return out.map(mapping).fillna(out)


def load_mortality(path: Path) -> pd.DataFrame:
    """
    Load observed mortality schedules by age, sex, and year. These schedules are the
    empirical basis for fitting Lee-Miller.
    """
    df = _load_csv(path)
    df = _rename_with_aliases(
        df,
        {
            "year": ["year", "anno", "time", "time_period", "tempo"],
            "age": ["age", "eta", "single_age", "eta_1", "età"],
            "sex": ["sex", "sesso", "gender"],
            "qx": ["qx", "probability_of_death", "probabilita_di_morte", "probabilita_morte", "q_x", "qx_per_1000"],
            "mx": ["mx", "mortality_rate", "death_rate", "m_x"],
        },
    )

    missing_base = [c for c in ["year", "age", "sex"] if c not in df.columns]
    if missing_base:
        raise ValueError(f"Mortality file is missing columns: {missing_base}")
    if "qx" not in df.columns and "mx" not in df.columns:
        raise ValueError("Mortality file must include either 'qx' or 'mx'.")

    keep = [c for c in ["year", "age", "sex", "qx", "mx"] if c in df.columns]
    df = df[keep].copy()

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["age"] = (
        df["age"].astype(str)
        .str.replace(" anni e più", "", regex=False)
        .str.replace(" anni", "", regex=False)
        .str.replace("+", "", regex=False)
        .str.replace("100 e oltre", "100", regex=False)
        .str.replace("100+", "100", regex=False)
    )
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["sex"] = standardize_sex(df["sex"])

    if "qx" in df.columns:
        df["qx"] = pd.to_numeric(df["qx"], errors="coerce")
    if "mx" in df.columns:
        df["mx"] = pd.to_numeric(df["mx"], errors="coerce")

    df = df.dropna(subset=["year", "age", "sex"])
    df["year"] = df["year"].astype(int)
    df["age"] = df["age"].astype(int)
    df = df[(df["age"] >= 0) & (df["age"] <= MAX_AGE)].copy()
    df = df[df["sex"].isin(["male", "female"])].copy()

    if "qx" not in df.columns:
        df["qx"] = qx_from_mx(df["mx"].to_numpy())
    else:
        if df["qx"].dropna().max() > 1.0:
            df["qx"] = df["qx"] / 1000.0
        df["qx"] = df["qx"].clip(QX_FLOOR, QX_CAP)

    if "mx" not in df.columns:
        df["mx"] = mx_from_qx(df["qx"].to_numpy())

    df = df.dropna(subset=["qx", "mx"])
    return df.sort_values(["sex", "year", "age"]).reset_index(drop=True)


def load_benchmarks(path: Optional[Path]) -> pd.DataFrame:
    """
    Load mortality benchmark data used later for validation, not for estimating our model.
    """
    if path is None:
        return pd.DataFrame()
    df = _load_csv(path)
    df = _rename_with_aliases(
        df,
        {
            "source": ["source", "producer", "ente", "benchmark"],
            "year": ["year", "anno", "time", "tempo"],
            "sex": ["sex", "sesso", "gender"],
            "e0": ["e0", "life_expectancy", "life_expectancy_at_birth", "speranza_di_vita", "ex"],
            "e0_lower": ["e0_lower", "lower", "lo", "lower_bound"],
            "e0_upper": ["e0_upper", "upper", "hi", "upper_bound"],
            "scenario": ["scenario", "variant"],
        },
    )
    needed = ["source", "year", "sex", "e0"]
    if any(c not in df.columns for c in needed):
        print(f"Benchmark file found but ignored because it lacks required columns {needed}: {path}")
        return pd.DataFrame()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["sex"] = standardize_sex(df["sex"])
    df["e0"] = pd.to_numeric(df["e0"], errors="coerce")
    for c in ["e0_lower", "e0_upper"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "scenario" not in df.columns:
        df["scenario"] = "central"
    return df.dropna(subset=["year", "sex", "e0"]).copy()


def _read_excel_fallback(path: Path, header=None) -> pd.DataFrame:
    """
    Read Excel sheets that may not have a clean header. This is used for external benchmark
    files with irregular layouts.
    """
    try:
        return pd.read_excel(path, header=header)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read Excel file {path}. Install openpyxl or save the file as CSV. Original error: {exc}"
        )


def load_wpp_qx_benchmarks() -> pd.DataFrame:
    """
    Load UN WPP age-specific mortality benchmarks, used to compare our qx schedules against
    an external source.
    """
    frames = []
    for sex, candidates in WPP_QX_FILE_CANDIDATES.items():
        path = _find_optional_file(candidates)
        if path is None:
            continue
        df = _read_excel_fallback(path, header=None)
        if df.shape[1] < 4:
            print(f"WPP qx file ignored because it has fewer than 4 columns: {path}")
            continue
        df = df.iloc[:, :4].copy()
        df.columns = ["country", "year", "age", "qx"]
        df["source"] = "UN WPP"
        df["sex"] = sex
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df["qx"] = pd.to_numeric(df["qx"], errors="coerce")
        df = df.dropna(subset=["year", "age", "qx"]).copy()
        df["year"] = df["year"].astype(int)
        df["age"] = df["age"].astype(int)
        df = df[(df["age"] >= 0) & (df["age"] <= MAX_AGE)].copy()
        if df["qx"].max() > 1.0:
            df["qx"] = df["qx"] / 1000.0
        df["qx"] = df["qx"].clip(QX_FLOOR, QX_CAP)
        frames.append(df[["source", "country", "sex", "year", "age", "qx"]])
        print(f"WPP qx file loaded: {path}")
    if not frames:
        return pd.DataFrame(columns=["source", "country", "sex", "year", "age", "qx"])
    return pd.concat(frames, ignore_index=True).sort_values(["sex", "year", "age"]).reset_index(drop=True)


def load_e0_other_benchmarks() -> pd.DataFrame:
    """
    Load external life-expectancy benchmarks, mainly for checking levels and trends of e0.
    """
    path = _find_optional_file(E0_OTHER_FILE_CANDIDATES)
    if path is None:
        return pd.DataFrame()
    df = _read_excel_fallback(path, header=0)
    df = _rename_with_aliases(
        df,
        {
            "source": ["source", "producer", "ente", "benchmark"],
            "sex": ["sex", "sesso", "gender"],
            "measure": ["measure", "scenario", "variant", "type"],
            "year": ["year", "anno", "time", "tempo"],
            "value": ["value", "e0", "life_expectancy", "life_expectancy_at_birth"],
        },
    )
    needed = ["source", "sex", "measure", "year", "value"]
    if any(c not in df.columns for c in needed):
        print(f"e0 benchmark file ignored because it lacks required columns {needed}: {path}")
        return pd.DataFrame()
    df = df[needed].copy()
    df["source"] = df["source"].astype(str).str.strip()
    df["sex"] = standardize_sex(df["sex"])
    df["measure"] = df["measure"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["source", "sex", "measure", "year", "value"]).copy()
    df["year"] = df["year"].astype(int)
    print(f"e0 benchmark file loaded: {path}")
    return df.sort_values(["source", "sex", "measure", "year"]).reset_index(drop=True)


def e0_benchmark_long_to_wide(e0_long: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape benchmark life expectancy data into a wide format that is easier to compare and
    plot.
    """
    if e0_long.empty:
        return pd.DataFrame()
    tmp = e0_long.copy()
    tmp["measure_clean"] = tmp["measure"].astype(str).str.strip().str.lower()
    tmp["measure_clean"] = tmp["measure_clean"].replace(
        {
            "median": "e0",
            "medium": "e0",
            "central": "e0",
            "mean": "e0",
            "lower90": "e0_lower90",
            "upper90": "e0_upper90",
            "lower95": "e0_lower95",
            "upper95": "e0_upper95",
        }
    )
    wide = tmp.pivot_table(index=["source", "sex", "year"], columns="measure_clean", values="value", aggfunc="mean").reset_index()
    wide.columns.name = None
    if "e0" not in wide.columns:
        candidates = [c for c in wide.columns if c not in ["source", "sex", "year"] and "median" in str(c).lower()]
        if candidates:
            wide["e0"] = wide[candidates[0]]
    if "e0" in wide.columns:
        wide["scenario"] = "central"
        cols = ["source", "sex", "year", "scenario", "e0"] + [c for c in ["e0_lower90", "e0_upper90", "e0_lower95", "e0_upper95"] if c in wide.columns]
        return wide[cols].sort_values(["source", "sex", "year"]).reset_index(drop=True)
    return pd.DataFrame()


def e0_from_wpp_qx(wpp_qx: pd.DataFrame) -> pd.DataFrame:
    """
    Compute life expectancy from WPP qx schedules so WPP can be compared on the same e0
    scale as our model.
    """
    if wpp_qx.empty:
        return pd.DataFrame()
    rows = []
    for (source, sex, year), g in wpp_qx.groupby(["source", "sex", "year"]):
        qx = g.sort_values("age")["qx"].to_numpy(dtype=float)
        rows.append({"source": f"{source} qx-derived", "sex": sex, "year": int(year), "scenario": "central", "e0": e0_from_qx(qx)})
    return pd.DataFrame(rows).sort_values(["source", "sex", "year"]).reset_index(drop=True)


def compare_qx_with_wpp(all_outputs: Dict[str, Dict[str, pd.DataFrame]], wpp_qx: pd.DataFrame) -> pd.DataFrame:
    """
    Compare our forecasted qx schedules with WPP schedules by age, sex, and year.
    """
    if wpp_qx.empty:
        return pd.DataFrame()
    model_qx = pd.concat([v["mean_schedule"] for v in all_outputs.values()], ignore_index=True)
    model_qx = model_qx[["model_version", "sex", "year", "age", "qx"]].rename(columns={"qx": "qx_model"})
    wpp = wpp_qx[["source", "sex", "year", "age", "qx"]].rename(columns={"source": "benchmark_source", "qx": "qx_benchmark"})
    comp = model_qx.merge(wpp, on=["sex", "year", "age"], how="inner")
    if comp.empty:
        return comp
    comp["qx_diff"] = comp["qx_model"] - comp["qx_benchmark"]
    comp["qx_ratio"] = comp["qx_model"] / comp["qx_benchmark"].replace(0, np.nan)
    comp["log_qx_diff"] = np.log(comp["qx_model"].clip(QX_FLOOR, QX_CAP)) - np.log(comp["qx_benchmark"].clip(QX_FLOOR, QX_CAP))
    return comp.sort_values(["model_version", "sex", "year", "age"]).reset_index(drop=True)


# ============================================================
# Lee-Carter / Lee-Miller fit
# ============================================================
@dataclass
class LeeMillerFit:
    age_params: pd.DataFrame
    kt_history: pd.DataFrame
    observed_vs_fitted: pd.DataFrame
    e0_history: pd.DataFrame


def fit_lee_miller_one_sex(df_sex: pd.DataFrame, sex_label: str, model_version: str) -> LeeMillerFit:
    """
    Fit the Lee-Miller model for one sex. The model separates the mortality age pattern from
    the period mortality trend.
    """
    years = sorted(df_sex["year"].astype(int).unique())
    ages = sorted(df_sex["age"].astype(int).unique())

    pivot = df_sex.pivot_table(index="age", columns="year", values="qx", aggfunc="mean")
    pivot = pivot.reindex(index=ages, columns=years)
    pivot = pivot.dropna(axis=0, how="any").dropna(axis=1, how="any")
    if pivot.shape[0] < 30 or pivot.shape[1] < 10:
        raise ValueError(f"Too little balanced data to fit Lee-Miller for {sex_label}, {model_version}.")

    ages = pivot.index.to_numpy(dtype=int)
    years = pivot.columns.to_numpy(dtype=int)
    qx_mat = np.clip(pivot.to_numpy(dtype=float), QX_FLOOR, QX_CAP)
    log_qx = np.log(qx_mat)

    ax = log_qx.mean(axis=1)
    centered = log_qx - ax[:, None]

    U, s, Vt = np.linalg.svd(centered, full_matrices=False)
    bx = U[:, 0]
    kt_raw = s[0] * Vt[0, :]

    scale = bx.sum()
    if abs(scale) < EPS:
        raise ValueError(f"Degenerate b_x for {sex_label}, {model_version}.")
    bx = bx / scale
    kt_raw = kt_raw * scale

    # Prefer lower k_t for lower mortality if possible.
    if np.nanmean(np.diff(kt_raw)) > 0:
        bx = -bx
        kt_raw = -kt_raw

    qx_raw = np.clip(np.exp(ax[:, None] + bx[:, None] * kt_raw[None, :]), QX_FLOOR, QX_CAP)

    observed_e0 = np.array([e0_from_qx(qx_mat[:, j]) for j in range(qx_mat.shape[1])], dtype=float)
    fitted_e0_raw = np.array([e0_from_qx(qx_raw[:, j]) for j in range(qx_raw.shape[1])], dtype=float)

    kt_adj = np.array(
        [lee_miller_adjust_kt(ax=ax, bx=bx, target_e0=observed_e0[j], kt_start=kt_raw[j]) for j in range(len(years))],
        dtype=float,
    )
    qx_adj = np.clip(np.exp(ax[:, None] + bx[:, None] * kt_adj[None, :]), QX_FLOOR, QX_CAP)
    fitted_e0_adj = np.array([e0_from_qx(qx_adj[:, j]) for j in range(qx_adj.shape[1])], dtype=float)

    age_params = pd.DataFrame({"model_version": model_version, "sex": sex_label, "age": ages, "ax": ax, "bx": bx})
    kt_history = pd.DataFrame(
        {
            "model_version": model_version,
            "sex": sex_label,
            "year": years,
            "kt_raw": kt_raw,
            "kt": kt_adj,
            "e0_observed": observed_e0,
            "e0_fitted_raw": fitted_e0_raw,
            "e0_fitted": fitted_e0_adj,
        }
    )
    e0_history = kt_history[["model_version", "sex", "year", "e0_observed", "e0_fitted"]].copy()

    ovf = pd.DataFrame(
        {
            "model_version": model_version,
            "age": np.repeat(ages, len(years)),
            "year": np.tile(years, len(ages)),
            "sex": sex_label,
            "qx_observed": qx_mat.reshape(-1, order="C"),
            "qx_fitted_raw": qx_raw.reshape(-1, order="C"),
            "qx_fitted": qx_adj.reshape(-1, order="C"),
        }
    ).sort_values(["sex", "year", "age"]).reset_index(drop=True)

    return LeeMillerFit(age_params=age_params, kt_history=kt_history, observed_vs_fitted=ovf, e0_history=e0_history)


# ============================================================
# Forecast helpers
# ============================================================
@dataclass
class ForecastResult:
    mean: pd.DataFrame
    simulations: pd.DataFrame
    model_summary: pd.DataFrame


def choose_rw_drift_model(series: pd.Series):
    """
    Fit the random-walk-with-drift model used for kt. This is the standard simple time-
    series choice for the Lee-Carter mortality index.
    """
    y = series.astype(float).dropna()
    model = ARIMA(y, order=ARIMA_ORDER_KT, trend="t").fit()
    return ARIMA_ORDER_KT, model


def simulate_arima(model, steps: int, n_sim: int, seed: int = RANDOM_SEED) -> np.ndarray:
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
        resid_sd = float(np.nanstd(model.resid)) if np.isfinite(np.nanstd(model.resid)) else 0.05
        se = np.repeat(max(resid_sd, 0.05), steps)
        return rs.normal(loc=mu, scale=se, size=(n_sim, steps))


def forecast_kt_one_sex(kt_history: pd.DataFrame, horizon_years: List[int], sex: str, model_version: str, n_sim: int) -> ForecastResult:
    """
    Forecast the mortality time index kt and simulate future kt paths. These paths drive
    uncertainty in future qx and e0.
    """
    y = kt_history.sort_values("year").set_index("year")["kt"].astype(float)
    order, model = choose_rw_drift_model(y)
    steps = len(horizon_years)

    fc = model.get_forecast(steps=steps)
    mean_df = pd.DataFrame({"model_version": model_version, "sex": sex, "year": horizon_years, "kt": np.asarray(fc.predicted_mean, dtype=float)})

    sims = simulate_arima(model, steps=steps, n_sim=n_sim, seed=RANDOM_SEED)
    sim_df = (
        pd.DataFrame(sims, columns=horizon_years)
        .assign(sim=np.arange(1, n_sim + 1), sex=sex, model_version=model_version)
        .melt(id_vars=["sim", "model_version", "sex"], var_name="year", value_name="kt")
        .sort_values(["sim", "year"])
        .reset_index(drop=True)
    )

    summary = pd.DataFrame(
        {
            "model_version": [model_version],
            "sex": [sex],
            "value": ["kt"],
            "arima_order": [str(order)],
            "aic": [float(model.aic)],
            "n_obs": [int(y.notna().sum())],
            "first_fit_year": [int(y.index.min())],
            "last_fit_year": [int(y.index.max())],
            "model": ["Random walk with drift via ARIMA(0,1,0)+drift"],
        }
    )
    return ForecastResult(mean=mean_df, simulations=sim_df, model_summary=summary)


def build_future_mortality_schedule(age_params: pd.DataFrame, kt_mean: pd.DataFrame, kt_sim: pd.DataFrame):
    """
    Convert forecasted kt paths back into full age-specific mortality schedules.
    """
    mean_rows = []
    for sex, gp in age_params.groupby("sex"):
        gp = gp.sort_values("age")
        ax = gp["ax"].to_numpy(dtype=float)
        bx = gp["bx"].to_numpy(dtype=float)
        ages = gp["age"].to_numpy(dtype=int)
        model_version = str(gp["model_version"].iloc[0])

        for _, row in kt_mean[kt_mean["sex"] == sex].iterrows():
            qx = reconstruct_qx(ax, bx, float(row["kt"]))
            mean_rows.append(
                pd.DataFrame({"model_version": model_version, "sex": sex, "year": int(row["year"]), "age": ages, "qx": qx, "px": 1.0 - qx, "mx": mx_from_qx(qx)})
            )
    mean_schedule = pd.concat(mean_rows, ignore_index=True)

    sim_rows = []
    for sex, gp in age_params.groupby("sex"):
        gp = gp.sort_values("age")
        ax = gp["ax"].to_numpy(dtype=float)
        bx = gp["bx"].to_numpy(dtype=float)
        ages = gp["age"].to_numpy(dtype=int)
        model_version = str(gp["model_version"].iloc[0])
        gk = kt_sim[kt_sim["sex"] == sex].copy()

        for (sim, year), rowg in gk.groupby(["sim", "year"]):
            kt = float(rowg["kt"].iloc[0])
            qx = reconstruct_qx(ax, bx, kt)
            sim_rows.append(
                pd.DataFrame({"sim": int(sim), "model_version": model_version, "sex": sex, "year": int(year), "age": ages, "qx": qx, "px": 1.0 - qx, "mx": mx_from_qx(qx)})
            )
    sim_schedule = pd.concat(sim_rows, ignore_index=True)
    return mean_schedule, sim_schedule


def life_expectancy_from_schedule(schedule_df: pd.DataFrame, has_sim: bool = False) -> pd.DataFrame:
    """
    Compute life expectancy from each forecasted mortality schedule, including simulated
    schedules when available.
    """
    rows = []
    grouping = ["model_version", "sex", "year"] if not has_sim else ["sim", "model_version", "sex", "year"]
    for keys, g in schedule_df.groupby(grouping):
        qx = g.sort_values("age")["qx"].to_numpy(dtype=float)
        e0 = e0_from_qx(qx)
        if has_sim:
            sim, model_version, sex, year = keys
            rows.append({"sim": int(sim), "model_version": model_version, "sex": sex, "year": int(year), "e0": e0})
        else:
            model_version, sex, year = keys
            rows.append({"model_version": model_version, "sex": sex, "year": int(year), "e0": e0})
    return pd.DataFrame(rows).sort_values(grouping).reset_index(drop=True)


def life_expectancy_intervals(e0_sim: pd.DataFrame, e0_mean: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize simulated life-expectancy paths into prediction intervals.
    """
    q = (
        e0_sim.groupby(["model_version", "sex", "year"])["e0"]
        .quantile([0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "p025", 0.1: "p10", 0.25: "p25", 0.5: "p50", 0.75: "p75", 0.9: "p90", 0.975: "p975"})
    )
    out = q.merge(e0_mean.rename(columns={"e0": "mean"}), on=["model_version", "sex", "year"], how="left")
    return out[["model_version", "sex", "year", "mean", "p025", "p10", "p25", "p50", "p75", "p90", "p975"]]


def merge_history_and_forecast_qx(hist_qx: pd.DataFrame, future_schedule_mean: pd.DataFrame, model_version: str) -> pd.DataFrame:
    """
    Combine observed qx and forecasted qx into one continuous output file.
    """
    hist = hist_qx[["sex", "year", "age", "qx", "mx"]].copy()
    hist["model_version"] = model_version
    hist["source"] = "observed"
    fut = future_schedule_mean[["model_version", "sex", "year", "age", "qx", "mx"]].copy()
    fut["source"] = "forecast"
    return pd.concat([hist, fut], ignore_index=True).sort_values(["model_version", "sex", "year", "age"]).reset_index(drop=True)


def fit_and_forecast_version(mort_df_full: pd.DataFrame, model_version: str, exclude_years: List[int], horizon_years: List[int]) -> Dict[str, pd.DataFrame]:
    """
    Run the full mortality pipeline for one model version, such as including or excluding
    Covid years.
    """
    out_dir = OUTPUT_ROOT / model_version
    out_dir.mkdir(parents=True, exist_ok=True)

    fit_df = mort_df_full[~mort_df_full["year"].isin(exclude_years)].copy()

    fits = []
    for sex, g in fit_df.groupby("sex"):
        fits.append(fit_lee_miller_one_sex(g.copy(), sex_label=sex, model_version=model_version))

    age_params = pd.concat([f.age_params for f in fits], ignore_index=True).sort_values(["sex", "age"]).reset_index(drop=True)
    kt_history = pd.concat([f.kt_history for f in fits], ignore_index=True).sort_values(["sex", "year"]).reset_index(drop=True)
    ovf = pd.concat([f.observed_vs_fitted for f in fits], ignore_index=True).sort_values(["sex", "year", "age"]).reset_index(drop=True)
    e0_history = pd.concat([f.e0_history for f in fits], ignore_index=True).sort_values(["sex", "year"]).reset_index(drop=True)

    age_params.to_csv(out_dir / "lee_miller_age_parameters.csv", index=False)
    kt_history.to_csv(out_dir / "lee_miller_time_index_history.csv", index=False)
    ovf.to_csv(out_dir / "mortality_fit_observed_vs_fitted.csv", index=False)
    e0_history.to_csv(out_dir / "life_expectancy_history.csv", index=False)

    mean_all, sim_all, model_info = [], [], []
    for sex, g in kt_history.groupby("sex"):
        fr = forecast_kt_one_sex(g, horizon_years=horizon_years, sex=sex, model_version=model_version, n_sim=N_SIM)
        mean_all.append(fr.mean)
        sim_all.append(fr.simulations)
        model_info.append(fr.model_summary)

    kt_mean = pd.concat(mean_all, ignore_index=True).sort_values(["sex", "year"]).reset_index(drop=True)
    kt_sim = pd.concat(sim_all, ignore_index=True).sort_values(["sim", "sex", "year"]).reset_index(drop=True)
    model_df = pd.concat(model_info, ignore_index=True)
    model_df["excluded_years"] = ",".join(map(str, exclude_years)) if exclude_years else "none"

    kt_mean.to_csv(out_dir / "lee_miller_time_index_forecast_mean.csv", index=False)
    kt_sim.to_csv(out_dir / "lee_miller_time_index_forecast_simulations.csv", index=False)
    model_df.to_csv(out_dir / "forecast_model_choices.csv", index=False)

    mean_schedule, sim_schedule = build_future_mortality_schedule(age_params, kt_mean, kt_sim)
    mean_schedule.to_csv(out_dir / "mortality_forecast_schedule_mean.csv", index=False)
    sim_schedule.to_csv(out_dir / "mortality_forecast_schedule_simulations.csv", index=False)

    mean_schedule[["model_version", "sex", "year", "age", "px"]].to_csv(out_dir / "survival_forecast_mean.csv", index=False)
    sim_schedule[["sim", "model_version", "sex", "year", "age", "px"]].to_csv(out_dir / "survival_forecast_simulations.csv", index=False)

    e0_mean = life_expectancy_from_schedule(mean_schedule, has_sim=False)
    e0_sim = life_expectancy_from_schedule(sim_schedule, has_sim=True)
    e0_int = life_expectancy_intervals(e0_sim, e0_mean)

    e0_mean.to_csv(out_dir / "life_expectancy_forecast_mean.csv", index=False)
    e0_sim.to_csv(out_dir / "life_expectancy_forecast_simulations.csv", index=False)
    e0_int.to_csv(out_dir / "life_expectancy_forecast_intervals.csv", index=False)

    full_qx = merge_history_and_forecast_qx(mort_df_full, mean_schedule, model_version=model_version)
    full_qx.to_csv(out_dir / "mortality_history_plus_forecast_mean.csv", index=False)

    return {
        "age_params": age_params,
        "kt_history": kt_history,
        "kt_mean": kt_mean,
        "kt_sim": kt_sim,
        "mean_schedule": mean_schedule,
        "sim_schedule": sim_schedule,
        "e0_history": e0_history,
        "e0_mean": e0_mean,
        "e0_sim": e0_sim,
        "e0_intervals": e0_int,
        "full_qx": full_qx,
        "model_info": model_df,
    }


def main():
    """
    Run the complete script from inputs to outputs.
    """
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    mortality_path = _find_input_file(MORTALITY_FILE_CANDIDATES)
    mort_df = load_mortality(mortality_path)

    last_observed_year = int(mort_df["year"].max())
    start_year = int(last_observed_year + 1) if FORECAST_START is None else int(FORECAST_START)
    horizon_years = list(range(start_year, FORECAST_END + 1))

    print(f"Mortality input: {mortality_path}")
    print(f"Observed years: {mort_df['year'].min()}-{mort_df['year'].max()}")
    print(f"Forecast years: {start_year}-{FORECAST_END}")

    all_outputs = {}
    for model_version, exclude_years in MODEL_RUNS.items():
        print(f"\nRunning model version: {model_version}")
        if exclude_years:
            print(f" - Excluding years from fit: {exclude_years}")
        all_outputs[model_version] = fit_and_forecast_version(mort_df, model_version, exclude_years, horizon_years)

    comparison_dir = OUTPUT_ROOT / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    pd.concat([v["e0_history"] for v in all_outputs.values()], ignore_index=True).to_csv(comparison_dir / "life_expectancy_history_all_models.csv", index=False)
    pd.concat([v["e0_mean"] for v in all_outputs.values()], ignore_index=True).to_csv(comparison_dir / "life_expectancy_forecast_mean_all_models.csv", index=False)
    pd.concat([v["e0_intervals"] for v in all_outputs.values()], ignore_index=True).to_csv(comparison_dir / "life_expectancy_forecast_intervals_all_models.csv", index=False)
    pd.concat([v["model_info"] for v in all_outputs.values()], ignore_index=True).to_csv(comparison_dir / "forecast_model_choices_all_models.csv", index=False)

    # This is the main qx file for the projection engine and for checks: age-by-age, year-by-year.
    pd.concat([v["full_qx"] for v in all_outputs.values()], ignore_index=True).to_csv(comparison_dir / "mortality_qx_history_plus_forecast_all_models.csv", index=False)

    benchmark_path = _find_optional_file(BENCHMARK_FILE_CANDIDATES)
    benchmarks_csv = load_benchmarks(benchmark_path)
    e0_other_long = load_e0_other_benchmarks()
    e0_other_wide = e0_benchmark_long_to_wide(e0_other_long)
    wpp_qx = load_wpp_qx_benchmarks()
    wpp_e0 = e0_from_wpp_qx(wpp_qx)

    if len(e0_other_long):
        e0_other_long.to_csv(comparison_dir / "life_expectancy_benchmarks_e0_long.csv", index=False)
    if len(e0_other_wide):
        e0_other_wide.to_csv(comparison_dir / "life_expectancy_benchmarks_e0_wide.csv", index=False)
    if len(wpp_qx):
        wpp_qx.to_csv(comparison_dir / "wpp_qx_benchmark.csv", index=False)
    if len(wpp_e0):
        wpp_e0.to_csv(comparison_dir / "wpp_life_expectancy_from_qx.csv", index=False)

    benchmark_frames = []
    if len(benchmarks_csv):
        benchmark_frames.append(benchmarks_csv)
        print(f"Benchmark e0 CSV file loaded: {benchmark_path}")
    if len(e0_other_wide):
        benchmark_frames.append(e0_other_wide)
    if len(wpp_e0):
        benchmark_frames.append(wpp_e0)

    if benchmark_frames:
        benchmark_all = pd.concat(benchmark_frames, ignore_index=True, sort=False)
        benchmark_all = benchmark_all.dropna(subset=["source", "sex", "year", "e0"]).copy()
        benchmark_all.to_csv(comparison_dir / "life_expectancy_benchmarks_e0.csv", index=False)
        print("e0 benchmarks included by source/sex:")
        print(benchmark_all.groupby(["source", "sex"]).size().to_string())
    else:
        print("No e0 benchmark file found. Add e0altri.xlsx or mortality_benchmarks_e0.csv to compare with ISTAT/UN/Eurostat.")

    qx_wpp_comp = compare_qx_with_wpp(all_outputs, wpp_qx)
    if len(qx_wpp_comp):
        qx_wpp_comp.to_csv(comparison_dir / "mortality_qx_forecast_vs_wpp.csv", index=False)

    print(f"\nDone. Output written to: {OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()
    
