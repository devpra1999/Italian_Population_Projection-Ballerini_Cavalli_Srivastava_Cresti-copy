"""
streamlit_projection_app.py

Self-contained Streamlit app for the all-at-once cohort-component projection.

This file intentionally does not import the external projection engine and does
not read precomputed files from output/.  It is designed for GitHub/Streamlit
deployment: the app reads the raw input files committed with the repository,
recreates the component forecasts internally, applies policy migration scenarios,
and then runs the cohort-component projection.

Raw files expected in the repository root:
    population.csv, or italiani.csv + stranieri.csv
    asfr.csv
    TFR.csv
    mortality.csv
    immigrati_input.csv
    emigrati_input.csv
    migration_altri.xlsx   optional, for migration benchmark display only
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

# Put cache folders before importing matplotlib, so Streamlit Cloud never tries
# to write Matplotlib/font caches in a protected home directory.
APP_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(APP_DIR / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(APP_DIR / ".cache"))

import matplotlib.pyplot as plt
import streamlit as st

try:
    from macro_simulations import (
        load_macro_data,
        load_model_data,
        compute_demo_ratios_from_projection,
        run_macro_simulations,
        plot_macro_italy,
        plot_oadr_italy,
        VAR_LABELS,
    )
    _MACRO_AVAILABLE = True
except ImportError:
    _MACRO_AVAILABLE = False


# =============================================================================
# Constants used throughout the projection
# =============================================================================

AGES = np.arange(0, 101, dtype=int)
SEXES = ["male", "female"]
CITIZENSHIPS = ["italiani", "stranieri"]
FLOW_NAMES = ["immigration", "emigration"]
SEX_RATIO_AT_BIRTH_MALE = 0.512
COVID_YEARS = {2020, 2021}
RANDOM_SEED = 123

SEX_TO_INDEX = {"male": 0, "female": 1}
CIT_TO_INDEX = {"italiani": 0, "stranieri": 1}
FLOW_TO_INDEX = {"immigration": 0, "emigration": 1}
SOURCE_COLORS = {
    "ISTAT": "#D55E00",
    "Istat": "#D55E00",
    "UN": "#009E73",
    "Eurostat": "#0072B2",
}
OUR_COLORS = {
    "male": "#4C78A8",
    "female": "#E45756",
}
LARGE_FIGSIZE = (13.5, 7.4)
PYRAMID_FIGSIZE = (10.5, 8.2)


# =============================================================================
# Scenario settings
# =============================================================================

@dataclass(frozen=True)
class Scenario:
    """Policy scenario applied after the baseline migration forecast is built."""

    name: str
    label: str
    extra_immigration_per_year: float = 0.0
    policy_start_year: int = 2026
    policy_end_year: int = 2075
    no_immigration: bool = False
    younger_with_children: bool = False
    child_share_of_extra: float = 0.25
    adult_min_age: int = 20
    adult_max_age: int = 34


def make_scenarios(
    policy_start_year: int,
    policy_end_year: int,
    extra_immigration_per_year: float,
    child_share_of_extra: float,
    adult_min_age: int,
    adult_max_age: int,
) -> list[Scenario]:
    """Create the migration scenarios shown in the Streamlit interface."""

    return [
        Scenario(
            name="baseline_migration",
            label="Baseline migration scenario",
        ),
        Scenario(
            name="no_immigration",
            label="No immigration scenario",
            no_immigration=True,
        ),
        Scenario(
            name="higher_total_immigration",
            label="Policy: higher total immigration only",
            extra_immigration_per_year=extra_immigration_per_year,
            policy_start_year=policy_start_year,
            policy_end_year=policy_end_year,
        ),
        Scenario(
            name="higher_immigration_younger_with_children",
            label="Policy: higher immigration + younger age structure + children",
            extra_immigration_per_year=extra_immigration_per_year,
            policy_start_year=policy_start_year,
            policy_end_year=policy_end_year,
            younger_with_children=True,
            child_share_of_extra=child_share_of_extra,
            adult_min_age=adult_min_age,
            adult_max_age=adult_max_age,
        ),
    ]


# =============================================================================
# Generic cleaning and reading helpers
# =============================================================================

def clean_colname(value: object) -> str:
    """Normalize column names from Italian/English files into snake_case."""

    text = str(value).strip().lower()
    for old, new in [("à", "a"), ("è", "e"), ("é", "e"), ("ì", "i"), ("ò", "o"), ("ù", "u")]:
        text = text.replace(old, new)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def normalize_label(value: object) -> str:
    """Normalize labels such as Maschi/Femmine/Totale for comparisons."""

    text = str(value).strip().lower()
    for old, new in [("à", "a"), ("è", "e"), ("é", "e"), ("ì", "i"), ("ò", "o"), ("ù", "u")]:
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text)


def standardize_sex(value: object) -> str:
    """Map Italian and English sex labels to the internal male/female labels."""

    text = normalize_label(value)
    if text in {"maschi", "maschio", "uomini", "uomo", "male", "m"}:
        return "male"
    if text in {"femmine", "femmina", "donne", "donna", "female", "f"}:
        return "female"
    return text


def standardize_citizenship(value: object) -> str:
    """Map citizenship/origin labels to italiani/stranieri."""

    text = normalize_label(value)
    if text in {"italia", "italiani", "italian", "italians"}:
        return "italiani"
    if text in {"paesi esteri", "stranieri", "foreign", "foreigners"}:
        return "stranieri"
    return text


def standardize_origin(value: object) -> str:
    """Keep migration origin labels in the form used by the input files."""

    text = normalize_label(value)
    if text in {"italia", "italiani"}:
        return "Italia"
    if text in {"paesi esteri", "stranieri"}:
        return "Paesi esteri"
    return str(value).strip()


def standardize_source(value: object) -> str:
    """Normalize official comparison source names."""

    text = str(value).strip()
    low = normalize_label(text)
    if low == "istat":
        return "ISTAT"
    if low == "un":
        return "UN"
    if low == "eurostat":
        return "Eurostat"
    return text


def is_total_label(value: object) -> bool:
    """Return True for total rows that should not be used as detail rows."""

    return normalize_label(value) in {"totale", "total", "all", "tutti", "tutte"}


def read_csv_flexible(path: Path) -> pd.DataFrame:
    """Read a CSV trying common separators."""

    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path.name}")
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    return pd.read_csv(path)


def rename_with_aliases(df: pd.DataFrame, aliases: Dict[str, Iterable[str]]) -> pd.DataFrame:
    """Rename accepted aliases to canonical names."""

    out = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")].copy()
    out.columns = [clean_colname(c) for c in out.columns]
    rename = {}
    for target, options in aliases.items():
        for option in options:
            key = clean_colname(option)
            if key in out.columns:
                rename[key] = target
                break
    return out.rename(columns=rename)


def parse_age(value: object) -> int:
    """Parse ages stored as integers or labels such as '100 anni e piu'."""

    if pd.isna(value):
        raise ValueError("Missing age")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and np.isfinite(value):
        return int(value)
    nums = [int(x) for x in re.findall(r"\d+", str(value))]
    if not nums:
        raise ValueError(f"Could not parse age: {value}")
    return min(nums[0], 100)


def parse_age_band(value: object) -> tuple[int, int]:
    """Parse broad migration age bands into lower and upper ages."""

    text = normalize_label(value)
    if is_total_label(text):
        raise ValueError("Total rows are not age bands")
    nums = [int(x) for x in re.findall(r"\d+", text)]
    if "fino" in text:
        return 0, nums[-1]
    if "piu" in text or "+" in text:
        return nums[0], 100
    if len(nums) >= 2:
        return nums[0], nums[1]
    if len(nums) == 1:
        return nums[0], nums[0]
    raise ValueError(f"Could not parse age band: {value}")


def summarize_simulations(values: np.ndarray, years: np.ndarray, scenario: Scenario, value_name: str) -> pd.DataFrame:
    """Convert a simulation array n_sim x n_years into mean and quantile bands."""

    quantiles = np.quantile(values, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], axis=0)
    return pd.DataFrame(
        {
            "scenario": scenario.name,
            "scenario_label": scenario.label,
            "year": years,
            "simulation_mean": values.mean(axis=0),
            "p05": quantiles[0],
            "p10": quantiles[1],
            "p25": quantiles[2],
            "p50": quantiles[3],
            "p75": quantiles[4],
            "p90": quantiles[5],
            "p95": quantiles[6],
            "indicator": value_name,
        }
    )


# =============================================================================
# Initial population
# =============================================================================

def load_initial_population(data_dir: Path, start_year: int) -> tuple[np.ndarray, pd.DataFrame, int]:
    """Load the initial population by citizenship, sex and single age."""

    population_path = data_dir / "population.csv"
    if population_path.exists():
        df = read_csv_flexible(population_path)
    else:
        frames = []
        for filename in ["italiani.csv", "stranieri.csv"]:
            frames.append(read_csv_flexible(data_dir / filename))
        df = pd.concat(frames, ignore_index=True)

    df = rename_with_aliases(
        df,
        {
            "sex": ["sex", "sesso"],
            "age": ["age", "eta"],
            "year": ["year", "anno"],
            "population": ["population", "popolazione", "pop", "value", "valore"],
            "citizenship": ["citizenship", "cittadinanza"],
        },
    )
    required = {"sex", "age", "year", "population", "citizenship"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Population file is missing columns: {sorted(missing)}")

    df["sex"] = df["sex"].map(standardize_sex)
    df["citizenship"] = df["citizenship"].map(standardize_citizenship)
    df["age"] = df["age"].map(parse_age).clip(0, 100)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["population"] = pd.to_numeric(df["population"], errors="coerce").fillna(0.0)
    df = df[df["citizenship"].isin(CITIZENSHIPS) & df["sex"].isin(SEXES)].dropna(subset=["year"]).copy()

    available_years = sorted(df["year"].astype(int).unique())
    initial_year = max([y for y in available_years if y <= start_year], default=max(available_years))
    df0 = df[df["year"].eq(initial_year)].copy()

    state = np.zeros((len(CITIZENSHIPS), len(SEXES), len(AGES)), dtype=float)
    for row in df0.itertuples(index=False):
        c = CIT_TO_INDEX[row.citizenship]
        s = SEX_TO_INDEX[row.sex]
        state[c, s, int(row.age)] += float(row.population)

    return state, df0, int(initial_year)


# =============================================================================
# Fertility from raw ASFR/TFR inputs
# =============================================================================

def load_fertility_observed(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read observed ASFR and TFR files."""

    asfr = rename_with_aliases(
        read_csv_flexible(data_dir / "asfr.csv"),
        {
            "citizenship": ["citizenship", "cittadinanza"],
            "age": ["age", "age_mother", "eta", "eta_madre"],
            "year": ["year", "anno"],
            "asfr": ["asfr", "fertility", "value", "valore"],
        },
    )
    tfr = rename_with_aliases(
        read_csv_flexible(data_dir / "TFR.csv"),
        {
            "citizenship": ["citizenship", "cittadinanza"],
            "year": ["year", "anno"],
            "tfr": ["tfr", "TFR", "value", "valore"],
        },
    )

    asfr["citizenship"] = asfr["citizenship"].map(standardize_citizenship)
    asfr["age"] = pd.to_numeric(asfr["age"], errors="coerce").astype("Int64")
    asfr["year"] = pd.to_numeric(asfr["year"], errors="coerce").astype("Int64")
    asfr["asfr"] = pd.to_numeric(asfr["asfr"], errors="coerce")
    asfr = asfr.dropna(subset=["age", "year", "asfr"]).copy()
    asfr["age"] = asfr["age"].astype(int)
    asfr["year"] = asfr["year"].astype(int)
    if asfr["asfr"].max() > 1.0:
        asfr["asfr"] = asfr["asfr"] / 1000.0

    tfr["citizenship"] = tfr["citizenship"].map(standardize_citizenship)
    tfr["year"] = pd.to_numeric(tfr["year"], errors="coerce").astype("Int64")
    tfr["tfr"] = pd.to_numeric(tfr["tfr"], errors="coerce")
    tfr = tfr.dropna(subset=["year", "tfr"]).copy()
    tfr["year"] = tfr["year"].astype(int)

    return asfr, tfr


def recent_asfr_shapes(asfr: pd.DataFrame, window: int = 5) -> np.ndarray:
    """Build fixed recent ASFR age-shapes, one for Italians and one for foreigners."""

    shapes = np.zeros((len(CITIZENSHIPS), len(AGES)), dtype=float)
    for citizenship in CITIZENSHIPS:
        g = asfr[asfr["citizenship"].eq(citizenship)].copy()
        recent_years = sorted(g["year"].unique())[-window:]
        recent = g[g["year"].isin(recent_years)].groupby("age", as_index=False)["asfr"].mean()
        vec = np.zeros(len(AGES), dtype=float)
        for row in recent.itertuples(index=False):
            if 0 <= int(row.age) <= 100:
                vec[int(row.age)] = max(float(row.asfr), 0.0)
        total = vec.sum()
        if total <= 0:
            vec[20:41] = 1.0
            total = vec.sum()
        shapes[CIT_TO_INDEX[citizenship], :] = vec / total
    return shapes


def damped_level_forecast(
    observed: pd.DataFrame,
    years: np.ndarray,
    value_col: str,
    lower: float,
    upper: float,
    damping: float = 0.94,
    slope_cap: float = 0.02,
    recent_window: int = 10,
) -> tuple[np.ndarray, float]:
    """Forecast a scalar demographic index with a damped recent linear trend."""

    obs = observed.sort_values("year").tail(recent_window).copy()
    y = obs[value_col].to_numpy(dtype=float)
    x = obs["year"].to_numpy(dtype=float)
    if len(y) >= 3:
        slope = np.polyfit(x - x.min(), y, 1)[0]
    else:
        slope = 0.0
    slope = float(np.clip(slope, -abs(slope_cap), abs(slope_cap)))
    last_value = float(y[-1])
    h = np.arange(len(years), dtype=float)
    path = last_value + slope * np.cumsum(damping ** h)
    diffs = np.diff(y)
    sigma = float(np.clip(np.std(diffs, ddof=1) if len(diffs) > 1 else 0.03, 0.015, 0.08))
    return np.clip(path, lower, upper), sigma


def mean_reverting_tfr_path(
    latest_value: float,
    years: np.ndarray,
    long_run_target: float,
    speed: float,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Build a smooth TFR path that converges toward a plausible long-run level.

    This mirrors the substantive idea of the fertility module better than a
    literal extrapolation of the latest negative slope.  Italy's observed TFR has
    fallen recently, but the baseline component forecast should not mechanically
    send Italian TFR to the numerical floor.  The path therefore starts from the
    latest observed level and softly approaches a low long-run level.
    """

    h = np.arange(len(years), dtype=float)
    path = long_run_target + (latest_value - long_run_target) * np.exp(-speed * h)
    return np.clip(path, lower, upper)


def simulate_ar1(n_sim: int, n_years: int, sigma: float, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Generate persistent AR(1)-style deviations for uncertainty paths."""

    out = np.zeros((n_sim, n_years), dtype=float)
    rho = float(np.clip(rho, -0.2, 0.95))
    innov = sigma * np.sqrt(max(1.0 - rho**2, 0.05))
    out[:, 0] = rng.normal(0.0, sigma, n_sim)
    for j in range(1, n_years):
        out[:, j] = rho * out[:, j - 1] + rng.normal(0.0, innov, n_sim)
    return out


def build_fertility_forecast(
    data_dir: Path,
    years: np.ndarray,
    n_sim: int,
    rng: np.random.Generator,
    tfr_increase_pct: float = 0.0,
) -> dict:
    """Forecast TFR and rebuild ASFR schedules from observed fertility inputs."""

    asfr_obs, tfr_obs = load_fertility_observed(data_dir)
    shapes = recent_asfr_shapes(asfr_obs)
    n_years = len(years)

    tfr_central = np.zeros((len(CITIZENSHIPS), n_years), dtype=float)
    tfr_sim = np.zeros((n_sim, len(CITIZENSHIPS), n_years), dtype=float)

    italians = tfr_obs[tfr_obs["citizenship"].eq("italiani")].sort_values("year").copy()
    foreign = tfr_obs[tfr_obs["citizenship"].eq("stranieri")].sort_values("year").copy()

    # The single-component fertility model forecasts the level and the age shape
    # separately.  Here, for Streamlit deployment, we reproduce the level idea
    # with transparent mean-reverting paths:
    # - Italians start from the latest observed TFR and move softly toward 1.20.
    # - Foreigners start from the latest observed TFR and move softly toward 1.95.
    # This avoids the coding mistake of extrapolating the last few declining
    # years indefinitely, which would push Italian TFR to an artificial 0.85 floor
    # and make the baseline population path look like a no-migration projection.
    italian_latest = float(italians["tfr"].iloc[-1])
    foreign_latest = float(foreign["tfr"].iloc[-1])
    italian_path = mean_reverting_tfr_path(
        italian_latest,
        years,
        long_run_target=1.20,
        speed=0.12,
        lower=0.95,
        upper=1.55,
    )
    italian_recent = italians["tfr"].tail(10).to_numpy(dtype=float)
    italian_sigma = float(np.clip(np.std(np.diff(italian_recent), ddof=1), 0.018, 0.060))
    tfr_central[CIT_TO_INDEX["italiani"], :] = italian_path
    italian_dev = simulate_ar1(n_sim, n_years, italian_sigma, 0.65, rng)
    tfr_sim[:, CIT_TO_INDEX["italiani"], :] = np.clip(italian_path[None, :] + italian_dev, 0.75, 1.80)

    foreign_path = mean_reverting_tfr_path(
        foreign_latest,
        years,
        long_run_target=1.95,
        speed=0.035,
        lower=1.20,
        upper=2.60,
    )
    gap_central = np.maximum(foreign_path - italian_path, 0.18)
    merged = foreign[["year", "tfr"]].merge(
        italians[["year", "tfr"]], on="year", how="inner", suffixes=("_foreign", "_italian")
    )
    if merged.empty or len(merged) < 4:
        gap_sigma = 0.04
    else:
        gap_sigma = float(np.clip(np.std(np.diff((merged["tfr_foreign"] - merged["tfr_italian"]).tail(10)), ddof=1), 0.02, 0.08))
    tfr_central[CIT_TO_INDEX["stranieri"], :] = foreign_path
    gap_dev = simulate_ar1(n_sim, n_years, gap_sigma, 0.60, rng)
    foreign_gap_sim = np.clip(gap_central[None, :] + gap_dev, 0.08, 1.30)
    tfr_sim[:, CIT_TO_INDEX["stranieri"], :] = np.clip(
        tfr_sim[:, CIT_TO_INDEX["italiani"], :] + foreign_gap_sim, 0.90, 2.60
    )

    # Optional policy/control lever: raise all forecast TFR schedules by x%.
    # The age pattern is unchanged; only the level of fertility changes.
    tfr_factor = 1.0 + float(tfr_increase_pct) / 100.0
    tfr_factor = max(tfr_factor, 0.0)
    tfr_central *= tfr_factor
    tfr_sim *= tfr_factor

    asfr_central = np.zeros((n_years, len(CITIZENSHIPS), len(AGES)), dtype=float)
    asfr_sim = np.zeros((n_sim, n_years, len(CITIZENSHIPS), len(AGES)), dtype=float)
    for c in range(len(CITIZENSHIPS)):
        asfr_central[:, c, :] = tfr_central[c, :, None] * shapes[c, :][None, :]
        asfr_sim[:, :, c, :] = tfr_sim[:, c, :, None] * shapes[c, :][None, :]

    tfr_rows = []
    for c_name, c in CIT_TO_INDEX.items():
        qs = np.quantile(tfr_sim[:, c, :], [0.10, 0.50, 0.90], axis=0)
        for j, year in enumerate(years):
            tfr_rows.append(
                {
                    "citizenship": c_name,
                    "year": int(year),
                    "tfr_mean": float(tfr_central[c, j]),
                    "tfr_p10": float(qs[0, j]),
                    "tfr_p50": float(qs[1, j]),
                    "tfr_p90": float(qs[2, j]),
                    "method": "mean_reverting_tfr_forecast_recent_asfr_shape",
                }
            )

    return {
        "asfr_central": asfr_central,
        "asfr_sim": asfr_sim,
        "tfr_summary": pd.DataFrame(tfr_rows),
        "asfr_observed": asfr_obs,
    }


# =============================================================================
# Mortality from raw qx input
# =============================================================================

def load_mortality_observed(data_dir: Path) -> pd.DataFrame:
    """Read observed mortality qx and standardize it to probabilities."""

    df = rename_with_aliases(
        read_csv_flexible(data_dir / "mortality.csv"),
        {
            "sex": ["sex", "sesso"],
            "age": ["age", "eta"],
            "year": ["year", "anno"],
            "qx": ["qx", "probability", "death_probability"],
            "mx": ["mx", "mortality_rate"],
        },
    )
    if "qx" not in df.columns and "mx" in df.columns:
        df["qx"] = 1.0 - np.exp(-pd.to_numeric(df["mx"], errors="coerce"))
    required = {"sex", "age", "year", "qx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Mortality file is missing columns: {sorted(missing)}")

    df["sex"] = df["sex"].map(standardize_sex)
    df["age"] = df["age"].map(parse_age).clip(0, 100)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["qx"] = pd.to_numeric(df["qx"], errors="coerce")
    df = df[df["sex"].isin(SEXES)].dropna(subset=["year", "qx"]).copy()
    df["year"] = df["year"].astype(int)
    if df["qx"].max() > 1.0:
        df["qx"] = df["qx"] / 1000.0
    df["qx"] = df["qx"].clip(1e-8, 0.999999)
    return df


def complete_mortality_panel(df: pd.DataFrame, sex: str) -> pd.DataFrame:
    """Create a complete year x age mortality panel for one sex."""

    g = df[df["sex"].eq(sex)].groupby(["year", "age"], as_index=False)["qx"].mean()
    years = np.arange(g["year"].min(), g["year"].max() + 1, dtype=int)
    index = pd.MultiIndex.from_product([years, AGES], names=["year", "age"])
    panel = g.set_index(["year", "age"]).reindex(index).reset_index()
    panel["sex"] = sex
    panel["qx"] = panel.groupby("year")["qx"].transform(lambda s: s.interpolate().ffill().bfill())
    panel["qx"] = panel.groupby("age")["qx"].transform(lambda s: s.interpolate().ffill().bfill())
    panel["qx"] = panel["qx"].clip(1e-8, 0.999999)
    return panel


def fit_forecast_mortality_for_sex(
    panel: pd.DataFrame,
    sex: str,
    years: np.ndarray,
    n_sim: int,
    rng: np.random.Generator,
    exclude_covid: bool,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Fit a Lee-Carter/Lee-Miller-style qx model and simulate survival."""

    fit_panel = panel[panel["sex"].eq(sex)].copy()
    if exclude_covid:
        fit_panel = fit_panel[~fit_panel["year"].isin(COVID_YEARS)].copy()
    fit_years = np.array(sorted(fit_panel["year"].unique()), dtype=int)
    matrix = (
        fit_panel.pivot(index="year", columns="age", values="qx")
        .reindex(index=fit_years, columns=AGES)
        .interpolate(axis=1)
        .ffill()
        .bfill()
        .to_numpy(dtype=float)
    )
    logqx = np.log(np.clip(matrix, 1e-8, 0.999999))
    ax = logqx.mean(axis=0)
    centered = logqx - ax[None, :]
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    bx = vt[0, :].copy()
    kt = u[:, 0] * s[0]
    if bx.sum() < 0:
        bx = -bx
        kt = -kt
    scale = bx.sum()
    if abs(scale) < 1e-12:
        scale = 1.0
    bx = bx / scale
    kt = kt * scale

    diffs = np.diff(kt)
    drift = float(np.mean(diffs[-10:])) if len(diffs) else 0.0
    drift = float(np.clip(drift, -8.0, 3.0))
    sigma = float(np.clip(np.std(diffs[-15:], ddof=1) if len(diffs) > 2 else 0.15, 0.03, 1.50))

    n_years = len(years)
    h = np.arange(1, n_years + 1, dtype=float)
    kt_central = kt[-1] + drift * h
    drift_draw = rng.normal(drift, sigma / max(np.sqrt(max(len(diffs), 1)), 1.0), n_sim)
    innovations = rng.normal(0.0, sigma, size=(n_sim, n_years))
    kt_sim = kt[-1] + np.cumsum(drift_draw[:, None] + innovations, axis=1)

    qx_central = np.exp(ax[None, :] + kt_central[:, None] * bx[None, :]).clip(1e-8, 0.999999)
    qx_sim = np.exp(ax[None, None, :] + kt_sim[:, :, None] * bx[None, None, :]).clip(1e-8, 0.999999)
    px_sim = 1.0 - qx_sim

    e0_rows = []
    for j, year in enumerate(years):
        e0_paths = np.array([life_expectancy_from_qx(qx_sim[i, j, :]) for i in range(n_sim)])
        qs = np.quantile(e0_paths, [0.10, 0.50, 0.90])
        e0_rows.append(
            {
                "sex": sex,
                "year": int(year),
                "e0_mean": float(life_expectancy_from_qx(qx_central[j, :])),
                "e0_p10": float(qs[0]),
                "e0_p50": float(qs[1]),
                "e0_p90": float(qs[2]),
                "method": "lee_carter_like_svd_random_walk_kt",
            }
        )

    return 1.0 - qx_central, px_sim, pd.DataFrame(e0_rows)


def life_expectancy_from_qx(qx: np.ndarray) -> float:
    """Compute period life expectancy at birth from single-age qx."""

    qx = np.asarray(qx, dtype=float).clip(1e-8, 0.999999)
    lx = np.empty(len(qx) + 1, dtype=float)
    lx[0] = 100_000.0
    for age in range(len(qx)):
        lx[age + 1] = lx[age] * (1.0 - qx[age])
    person_years = 0.5 * (lx[:-1] + lx[1:])
    person_years[-1] = lx[-2] / max(qx[-1], 1e-8)
    return float(person_years.sum() / lx[0])


def build_mortality_forecast(
    data_dir: Path,
    years: np.ndarray,
    n_sim: int,
    rng: np.random.Generator,
    mortality_version: str,
) -> dict:
    """Build survival probabilities by sex, age, year and simulation path."""

    observed = load_mortality_observed(data_dir)
    exclude_covid = mortality_version == "excluding_covid"
    px_central = np.zeros((len(years), len(SEXES), len(AGES)), dtype=float)
    px_sim = np.zeros((n_sim, len(years), len(SEXES), len(AGES)), dtype=float)
    e0_frames = []
    for sex in SEXES:
        panel = complete_mortality_panel(observed, sex)
        central, sim, e0 = fit_forecast_mortality_for_sex(panel, sex, years, n_sim, rng, exclude_covid)
        s = SEX_TO_INDEX[sex]
        px_central[:, s, :] = central
        px_sim[:, :, s, :] = sim
        e0_frames.append(e0)
    return {"px_central": px_central, "px_sim": px_sim, "e0_summary": pd.concat(e0_frames, ignore_index=True)}


# =============================================================================
# Migration from raw broad-age input files
# =============================================================================

def read_migration_file(data_dir: Path, filename: str, flow: str) -> pd.DataFrame:
    """Read immigration or emigration broad age-band rows."""

    df = rename_with_aliases(
        read_csv_flexible(data_dir / filename),
        {
            "origin": ["origine", "origin", "citizenship", "cittadinanza"],
            "sex": ["sex", "sesso"],
            "age_band": ["age", "eta", "classe_eta", "classi_eta"],
            "year": ["year", "anno"],
            "count": ["count", "value", "valore", "immigrati", "emigrati"],
        },
    )
    required = {"origin", "sex", "age_band", "year", "count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{filename} is missing columns: {sorted(missing)}")

    df["origin"] = df["origin"].map(standardize_origin)
    df["sex"] = df["sex"].map(standardize_sex)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["count"] = pd.to_numeric(df["count"], errors="coerce")
    df = df.dropna(subset=["year", "count"]).copy()
    df = df[
        df["origin"].isin(["Italia", "Paesi esteri"])
        & df["sex"].isin(SEXES)
        & ~df["age_band"].map(is_total_label)
    ].copy()
    parsed = df["age_band"].map(parse_age_band)
    df["age_min"] = parsed.map(lambda x: x[0])
    df["age_max"] = parsed.map(lambda x: x[1])
    df["citizenship"] = df["origin"].map({"Italia": "italiani", "Paesi esteri": "stranieri"})
    df["flow"] = flow
    df["year"] = df["year"].astype(int)
    return df[["flow", "year", "origin", "citizenship", "sex", "age_band", "age_min", "age_max", "count"]]


def load_migration_observed(data_dir: Path) -> pd.DataFrame:
    """Load both migration input files and combine them."""

    immigration = read_migration_file(data_dir, "immigrati_input.csv", "immigration")
    emigration = read_migration_file(data_dir, "emigrati_input.csv", "emigration")
    return pd.concat([immigration, emigration], ignore_index=True)


def fit_log_linear(year: np.ndarray, value: np.ndarray) -> dict:
    """Fit log(value) = intercept + slope * year_index."""

    y = np.log(np.clip(value.astype(float), 1.0, None))
    x = year.astype(float) - float(year.min())
    xmat = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(xmat, y, rcond=None)[0]
    fitted = xmat @ beta
    resid = y - fitted
    dof = max(len(y) - 2, 1)
    sigma = float(np.sqrt(np.sum(resid**2) / dof))
    return {"intercept": float(beta[0]), "slope": float(beta[1]), "sigma": sigma, "residuals": resid}


def damped_log_path(start_value: float, slope: float, years: np.ndarray, damping: float) -> np.ndarray:
    """Project a total flow using a damped annual log slope."""

    h = np.arange(len(years), dtype=float)
    cumulative = np.cumsum(damping**h)
    return np.exp(np.log(max(start_value, 1.0)) + slope * cumulative)


def choose_years_available(observed_years: Iterable[int], preferred: Iterable[int], fallback_n: int) -> list[int]:
    """Keep preferred years if available, otherwise use the latest fallback_n years."""

    available = sorted(set(int(y) for y in observed_years))
    chosen = [int(y) for y in preferred if int(y) in available]
    if len(chosen) >= 2:
        return chosen
    return available[-fallback_n:]


def forecast_migration_totals(
    observed_bands: pd.DataFrame,
    years: np.ndarray,
    n_sim: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Forecast total immigration and emigration with damped log trends."""

    totals = observed_bands.groupby(["year", "flow"], as_index=False)["count"].sum()
    wide = totals.pivot(index="year", columns="flow", values="count").reset_index().fillna(0.0)
    observed_years = wide["year"].astype(int).unique()

    recent_years = choose_years_available(observed_years, [2022, 2023, 2024, 2025], 4)
    imm_trend_years = choose_years_available(observed_years, range(2007, 2020), 12)
    emi_trend_years = choose_years_available(
        observed_years, [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025], 12
    )

    n_years = len(years)
    central = np.zeros((len(FLOW_NAMES), n_years), dtype=float)
    sim = np.zeros((n_sim, len(FLOW_NAMES), n_years), dtype=float)

    settings = {
        "immigration": (imm_trend_years, 0.90, 0.035),
        "emigration": (emi_trend_years, 0.93, 0.025),
    }
    rows = []
    recent = wide[wide["year"].isin(recent_years)]
    for flow in FLOW_NAMES:
        trend_years, damping, slope_cap = settings[flow]
        trend = wide[wide["year"].isin(trend_years)].copy()
        if len(trend) < 5:
            trend = wide.tail(min(len(wide), 8)).copy()
        fit = fit_log_linear(trend["year"].to_numpy(), trend[flow].to_numpy())
        slope = float(np.clip(fit["slope"], -abs(slope_cap), abs(slope_cap)))
        sigma = float(np.clip(fit["sigma"], 0.035, 0.25))
        start = float(recent[flow].mean())
        path = damped_log_path(start, slope, years, damping)
        f = FLOW_TO_INDEX[flow]
        central[f, :] = path

        slope_draw = rng.normal(slope, min(sigma / 10.0, 0.010), n_sim)
        slope_draw = np.clip(slope_draw, -abs(slope_cap) - 0.005, abs(slope_cap) + 0.005)
        start_draw = rng.normal(np.log(max(start, 1.0)), sigma, n_sim)
        eps = simulate_ar1(n_sim, n_years, sigma, 0.55, rng)
        cumulative = np.cumsum(damping ** np.arange(n_years, dtype=float))
        sim[:, f, :] = np.exp(start_draw[:, None] + slope_draw[:, None] * cumulative[None, :] + eps)

        for j, year in enumerate(years):
            rows.append(
                {
                    "flow": flow,
                    "year": int(year),
                    "mean": float(path[j]),
                    "trend_years": ",".join(map(str, trend_years)),
                    "recent_level_years": ",".join(map(str, recent_years)),
                    "method": "damped_log_trend_total_flow",
                }
            )

    return pd.DataFrame(rows), sim


def rogers_castro_template(flow: str) -> np.ndarray:
    """Smooth age template used as a demographic prior for migration profiles."""

    age = AGES.astype(float)
    if flow == "immigration":
        profile = (
            0.035
            + 0.30 * np.exp(-0.5 * ((age - 5.0) / 8.0) ** 2)
            + 1.00 * np.exp(-0.5 * ((age - 29.0) / 8.5) ** 2)
            + 0.32 * np.exp(-0.5 * ((age - 40.0) / 11.0) ** 2)
            + 0.10 * np.exp(-0.5 * ((age - 66.0) / 8.0) ** 2)
        )
    else:
        profile = (
            0.030
            + 0.24 * np.exp(-0.5 * ((age - 6.0) / 8.0) ** 2)
            + 1.00 * np.exp(-0.5 * ((age - 31.0) / 9.5) ** 2)
            + 0.25 * np.exp(-0.5 * ((age - 43.0) / 12.0) ** 2)
            + 0.12 * np.exp(-0.5 * ((age - 66.0) / 8.0) ** 2)
        )
    return np.clip(profile, 1e-12, None)


def smooth_rake(profile: np.ndarray, band_targets: list[tuple[int, int, float]], iterations: int = 80) -> np.ndarray:
    """Smooth a single-age vector while preserving every broad-age band total."""

    out = np.asarray(profile, dtype=float).copy()
    kernel = np.array([0.25, 0.50, 0.25])
    for _ in range(iterations):
        smoothed = out.copy()
        smoothed[1:-1] = kernel[0] * out[:-2] + kernel[1] * out[1:-1] + kernel[2] * out[2:]
        out = 0.70 * out + 0.30 * smoothed
        for age_min, age_max, target in band_targets:
            idx = np.arange(age_min, age_max + 1)
            current = out[idx].sum()
            if current > 0:
                out[idx] *= target / current
            else:
                out[idx] = target / len(idx)
    return out


def component_shares(observed_bands: pd.DataFrame) -> pd.DataFrame:
    """Estimate recent broad-age/origin/sex shares for each gross flow."""

    years = sorted(observed_bands["year"].unique())[-4:]
    recent = observed_bands[observed_bands["year"].isin(years)].copy()
    comp = (
        recent.groupby(["flow", "origin", "citizenship", "sex", "age_min", "age_max"], as_index=False)["count"]
        .sum()
    )
    totals = comp.groupby("flow", as_index=False)["count"].sum().rename(columns={"count": "flow_total"})
    comp = comp.merge(totals, on="flow", how="left")
    comp["share"] = np.where(comp["flow_total"] > 0, comp["count"] / comp["flow_total"], 0.0)
    return comp


def population_age_weights(initial_state: np.ndarray, citizenship: str, sex: str) -> np.ndarray:
    """Get current population age weights for one citizenship and sex."""

    vec = initial_state[CIT_TO_INDEX[citizenship], SEX_TO_INDEX[sex], :].astype(float).copy()
    if vec.sum() <= 0:
        vec[:] = 1.0
    return vec / vec.sum()


def expand_migration_mean_to_single_age(
    initial_state: np.ndarray,
    observed_bands: pd.DataFrame,
    total_mean: pd.DataFrame,
    years: np.ndarray,
) -> np.ndarray:
    """Allocate total migration forecasts to citizenship, sex and single age."""

    shares = component_shares(observed_bands)
    mean_totals = total_mean.pivot(index="year", columns="flow", values="mean").reindex(years).ffill().bfill()
    out = np.zeros((len(FLOW_NAMES), len(years), len(CITIZENSHIPS), len(SEXES), len(AGES)), dtype=float)

    for flow in FLOW_NAMES:
        template = rogers_castro_template(flow)
        f = FLOW_TO_INDEX[flow]
        flow_shares = shares[shares["flow"].eq(flow)].copy()
        for y_idx, year in enumerate(years):
            flow_total = float(mean_totals.loc[int(year), flow])
            for (citizenship, sex), comp in flow_shares.groupby(["citizenship", "sex"]):
                vec = np.zeros(len(AGES), dtype=float)
                band_targets = []
                pop_weights = population_age_weights(initial_state, citizenship, sex)
                for row in comp.itertuples(index=False):
                    target = flow_total * float(row.share)
                    idx = np.arange(int(row.age_min), int(row.age_max) + 1)
                    if int(row.age_min) == 0 and int(row.age_max) <= 17:
                        raw = 0.85 * pop_weights[idx] + 0.15 * template[idx] / template[idx].sum()
                    else:
                        raw = pop_weights[idx] * template[idx]
                    raw = np.clip(raw, 1e-12, None)
                    vec[idx] = target * raw / raw.sum()
                    band_targets.append((int(row.age_min), int(row.age_max), target))
                vec = smooth_rake(vec, band_targets)
                out[f, y_idx, CIT_TO_INDEX[citizenship], SEX_TO_INDEX[sex], :] = vec
    return out


def build_migration_forecast(
    data_dir: Path,
    years: np.ndarray,
    n_sim: int,
    rng: np.random.Generator,
    initial_state: np.ndarray,
) -> dict:
    """Build migration totals, simulations and single-age allocations."""

    observed = load_migration_observed(data_dir)
    total_mean, total_sim = forecast_migration_totals(observed, years, n_sim, rng)
    single_age_mean = expand_migration_mean_to_single_age(initial_state, observed, total_mean, years)

    single_age_sim = np.zeros((n_sim, len(FLOW_NAMES), len(years), len(CITIZENSHIPS), len(SEXES), len(AGES)), dtype=float)
    for f, flow in enumerate(FLOW_NAMES):
        central_total = single_age_mean[f].sum(axis=(1, 2, 3))
        for j in range(len(years)):
            ratio = total_sim[:, f, j] / max(central_total[j], 1.0)
            single_age_sim[:, f, j, :, :, :] = single_age_mean[f, j, :, :, :][None, :, :, :] * ratio[:, None, None, None]

    return {
        "observed_bands": observed,
        "total_mean": total_mean,
        "total_sim": total_sim,
        "single_age_mean": single_age_mean,
        "single_age_sim": single_age_sim,
    }


# =============================================================================
# Policy migration modifiers
# =============================================================================

def baseline_immigration_shares(immigration: np.ndarray) -> np.ndarray:
    """Return age-sex-citizenship shares for baseline immigration."""

    total = immigration.sum()
    if total <= 0:
        shares = np.zeros_like(immigration)
        shares[CIT_TO_INDEX["stranieri"], :, 20:35] = 1.0
        return shares / shares.sum()
    return immigration / total


def younger_policy_extra_distribution(
    initial_state: np.ndarray,
    child_share: float,
    adult_min_age: int,
    adult_max_age: int,
) -> np.ndarray:
    """Distribute extra policy immigrants among young adults and children."""

    dist = np.zeros((len(CITIZENSHIPS), len(SEXES), len(AGES)), dtype=float)
    foreign = CIT_TO_INDEX["stranieri"]

    adult_min_age = int(np.clip(adult_min_age, 0, 100))
    adult_max_age = int(np.clip(adult_max_age, adult_min_age, 100))
    adult_ages = np.arange(adult_min_age, adult_max_age + 1)
    center = (adult_min_age + adult_max_age) / 2.0
    denominator = max(center - adult_ages.min(), adult_ages.max() - center, 1.0)
    adult_weights = 1.0 - np.abs(adult_ages - center) / denominator
    adult_weights = np.clip(adult_weights, 0.15, None)
    adult_weights = adult_weights / adult_weights.sum()

    child_ages = np.arange(0, 18)
    for sex in SEXES:
        s = SEX_TO_INDEX[sex]
        child_pop = initial_state[foreign, s, child_ages].astype(float)
        if child_pop.sum() <= 0:
            child_weights = np.ones(len(child_ages)) / len(child_ages)
        else:
            child_weights = child_pop / child_pop.sum()
        dist[foreign, s, adult_ages] += (1.0 - child_share) * 0.5 * adult_weights
        dist[foreign, s, child_ages] += child_share * 0.5 * child_weights
    return dist / dist.sum()


def apply_migration_scenario(
    base_immigration: np.ndarray,
    base_emigration: np.ndarray,
    scenario: Scenario,
    year: int,
    initial_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the selected scenario to baseline immigration only."""

    immigration = base_immigration.copy()
    emigration = base_emigration.copy()
    if scenario.no_immigration:
        immigration[:] = 0.0
        return immigration, emigration
    if not (scenario.policy_start_year <= year <= scenario.policy_end_year):
        return immigration, emigration

    base_total = float(immigration.sum())
    extra = max(float(scenario.extra_immigration_per_year), 0.0)
    if extra <= 0:
        return immigration, emigration

    if scenario.younger_with_children:
        dist = younger_policy_extra_distribution(
            initial_state,
            scenario.child_share_of_extra,
            scenario.adult_min_age,
            scenario.adult_max_age,
        )
    else:
        dist = baseline_immigration_shares(immigration)
    immigration = immigration + extra * dist
    return immigration, emigration


# =============================================================================
# Cohort-component projection engine
# =============================================================================

def run_projection(
    data_dir: Path,
    start_year: int,
    end_year: int,
    n_sim: int,
    mortality_version: str,
    scenarios: list[Scenario],
    tfr_increase_pct: float = 0.0,
) -> dict:
    """Run the complete all-at-once cohort-component projection."""

    rng = np.random.default_rng(RANDOM_SEED)
    transition_years = np.arange(start_year, end_year, dtype=int)
    state_years = np.arange(start_year, end_year + 1, dtype=int)

    initial_state, initial_population_df, initial_population_year = load_initial_population(data_dir, start_year)
    fertility = build_fertility_forecast(data_dir, transition_years, n_sim, rng, tfr_increase_pct)
    mortality = build_mortality_forecast(data_dir, transition_years, n_sim, rng, mortality_version)
    migration = build_migration_forecast(data_dir, transition_years, n_sim, rng, initial_state)

    n_scenarios = len(scenarios)
    n_state_years = len(state_years)
    n_transition_years = len(transition_years)

    total_pop = np.zeros((n_scenarios, n_sim, n_state_years), dtype=float)
    births = np.zeros((n_scenarios, n_sim, n_transition_years), dtype=float)
    oadr = np.zeros((n_scenarios, n_sim, n_state_years), dtype=float)
    pop_age_sex = np.zeros((n_scenarios, n_sim, n_state_years, len(SEXES), len(AGES)), dtype=float)
    migration_totals = np.zeros((n_scenarios, n_sim, n_transition_years, 3), dtype=float)
    total_tfr = np.zeros((n_scenarios, n_sim, n_transition_years), dtype=float)

    for scen_idx, scenario in enumerate(scenarios):
        for sim_idx in range(n_sim):
            state = initial_state.copy()
            save_state_arrays(state, scen_idx, sim_idx, 0, total_pop, oadr, pop_age_sex)

            for y_idx, year in enumerate(transition_years):
                px = mortality["px_sim"][sim_idx, y_idx, :, :]
                asfr = fertility["asfr_sim"][sim_idx, y_idx, :, :]
                base_immigration = migration["single_age_sim"][sim_idx, FLOW_TO_INDEX["immigration"], y_idx, :, :, :]
                base_emigration = migration["single_age_sim"][sim_idx, FLOW_TO_INDEX["emigration"], y_idx, :, :, :]
                immigration, emigration = apply_migration_scenario(
                    base_immigration, base_emigration, scenario, int(year), initial_state
                )
                total_tfr[scen_idx, sim_idx, y_idx] = compute_total_tfr_from_state(state, asfr)

                next_state, total_births, applied_emigration = project_one_year(state, px, asfr, immigration, emigration)
                migration_totals[scen_idx, sim_idx, y_idx, 0] = immigration.sum()
                migration_totals[scen_idx, sim_idx, y_idx, 1] = applied_emigration.sum()
                migration_totals[scen_idx, sim_idx, y_idx, 2] = immigration.sum() - applied_emigration.sum()
                births[scen_idx, sim_idx, y_idx] = total_births

                state = next_state
                save_state_arrays(state, scen_idx, sim_idx, y_idx + 1, total_pop, oadr, pop_age_sex)

    population_total_summary = pd.concat(
        [summarize_simulations(total_pop[i], state_years, scenarios[i], "total_population") for i in range(n_scenarios)],
        ignore_index=True,
    )
    births_summary = pd.concat(
        [summarize_simulations(births[i], transition_years, scenarios[i], "births") for i in range(n_scenarios)],
        ignore_index=True,
    )
    oadr_summary = pd.concat(
        [summarize_simulations(oadr[i], state_years, scenarios[i], "old_age_dependency_ratio") for i in range(n_scenarios)],
        ignore_index=True,
    )
    migration_summary = summarize_migration_totals(migration_totals, transition_years, scenarios)
    total_tfr_summary = pd.concat(
        [summarize_simulations(total_tfr[i], transition_years, scenarios[i], "total_tfr") for i in range(n_scenarios)],
        ignore_index=True,
    )
    population_by_age_mean = make_population_by_age_mean(pop_age_sex, state_years, scenarios)

    return {
        "population_total_summary": population_total_summary,
        "births_summary": births_summary,
        "old_age_dependency_summary": oadr_summary,
        "migration_summary": migration_summary,
        "total_tfr_summary": total_tfr_summary,
        "population_by_age_mean": population_by_age_mean,
        "tfr_summary": fertility["tfr_summary"],
        "e0_summary": mortality["e0_summary"],
        "initial_population_year": initial_population_year,
        "initial_population_df": initial_population_df,
        "mortality_version": mortality_version,
    }


def project_one_year(
    state: np.ndarray,
    px: np.ndarray,
    asfr: np.ndarray,
    immigration: np.ndarray,
    emigration: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Advance the population one year with survival, migration and births."""

    survivors = state * px[None, :, :]
    applied_emigration = np.minimum(emigration, survivors)
    post_migration = survivors - applied_emigration + immigration

    female = SEX_TO_INDEX["female"]
    net_female_migration = immigration[:, female, :] - applied_emigration[:, female, :]
    female_exposure = np.clip(state[:, female, :] + 0.5 * net_female_migration, 0.0, None)
    births_by_citizenship = (female_exposure * asfr).sum(axis=1)
    total_births = float(births_by_citizenship.sum())

    next_state = np.zeros_like(state)
    next_state[:, :, 1:] += post_migration[:, :, :-1]
    next_state[:, :, 100] += post_migration[:, :, 100]
    next_state[:, SEX_TO_INDEX["male"], 0] = births_by_citizenship * SEX_RATIO_AT_BIRTH_MALE
    next_state[:, SEX_TO_INDEX["female"], 0] = births_by_citizenship * (1.0 - SEX_RATIO_AT_BIRTH_MALE)
    return next_state, total_births, applied_emigration


def compute_total_tfr_from_state(state: np.ndarray, asfr: np.ndarray) -> float:
    """Compute total-population TFR from citizenship-specific ASFR schedules.

    The input fertility model gives ASFR by citizenship.  For a single total TFR
    for Italy, we combine those ASFRs using the current age-specific distribution
    of women by citizenship.  This is more precise than applying one fixed
    Italian/foreign weight to both groups at all reproductive ages.
    """

    female = SEX_TO_INDEX["female"]
    female_by_cit_age = state[:, female, :].astype(float)
    total_female_age = female_by_cit_age.sum(axis=0)
    total_asfr_by_age = np.divide(
        (female_by_cit_age * asfr).sum(axis=0),
        total_female_age,
        out=np.zeros_like(total_female_age, dtype=float),
        where=total_female_age > 0,
    )
    return float(total_asfr_by_age.sum())


def save_state_arrays(
    state: np.ndarray,
    scen_idx: int,
    sim_idx: int,
    year_idx: int,
    total_pop: np.ndarray,
    oadr: np.ndarray,
    pop_age_sex: np.ndarray,
) -> None:
    """Store total population, old-age dependency and age-sex population."""

    total_pop[scen_idx, sim_idx, year_idx] = state.sum()
    working = state[:, :, 15:65].sum()
    older = state[:, :, 65:].sum()
    oadr[scen_idx, sim_idx, year_idx] = 100.0 * older / working if working > 0 else np.nan
    pop_age_sex[scen_idx, sim_idx, year_idx, :, :] = state.sum(axis=0)


def summarize_migration_totals(migration_totals: np.ndarray, years: np.ndarray, scenarios: list[Scenario]) -> pd.DataFrame:
    """Summarize immigration, emigration and net migration by scenario."""

    frames = []
    labels = ["immigration", "emigration", "net_migration"]
    for i, scenario in enumerate(scenarios):
        for k, label in enumerate(labels):
            frames.append(summarize_simulations(migration_totals[i, :, :, k], years, scenario, label))
    return pd.concat(frames, ignore_index=True)


def make_population_by_age_mean(pop_age_sex: np.ndarray, years: np.ndarray, scenarios: list[Scenario]) -> pd.DataFrame:
    """Create a compact mean population-by-age table for pyramid plots."""

    rows = []
    mean_array = pop_age_sex.mean(axis=1)
    for i, scenario in enumerate(scenarios):
        for y_idx, year in enumerate(years):
            for sex in SEXES:
                s = SEX_TO_INDEX[sex]
                for age in AGES:
                    rows.append(
                        {
                            "scenario": scenario.name,
                            "scenario_label": scenario.label,
                            "year": int(year),
                            "sex": sex,
                            "age": int(age),
                            "population": float(mean_array[i, y_idx, s, int(age)]),
                        }
                    )
    return pd.DataFrame(rows)


# =============================================================================
# Optional benchmark loader
# =============================================================================

def load_migration_benchmarks(data_dir: Path) -> pd.DataFrame:
    """Load migration_altri.xlsx if available for component diagnostics."""

    path = data_dir / "migration_altri.xlsx"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_excel(path)
    except Exception:
        return pd.DataFrame()
    df = rename_with_aliases(
        df,
        {
            "source": ["source"],
            "flow": ["emi_imm_net", "emi/imm/net", "flow"],
            "measure": ["measure"],
            "year": ["year"],
            "value": ["value"],
        },
    )
    if {"source", "flow", "measure", "year", "value"} - set(df.columns):
        return pd.DataFrame()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["year", "value"]).copy()
    # The workbook stores values in thousands, so convert to persons.
    if df["value"].abs().median() < 10_000:
        df["value"] = df["value"] * 1000.0
    return df


def load_official_population_benchmarks(data_dir: Path) -> pd.DataFrame:
    """Placeholder loader for official total-population projection comparisons.

    TODO after downloading comparison data:
    Create a CSV named official_population_projection_comparisons.csv in the
    repository root with columns:
        source, scenario, year, population

    Expected sources can include:
        Istat, Eurostat, UN, PopulationPyramids.net

    Once the file exists, the total-population figure will automatically add
    those official series. Until then this function returns an empty table.
    """

    path = data_dir / "official_population_projection_comparisons.csv"
    if not path.exists():
        return pd.DataFrame(columns=["source", "scenario", "year", "population"])
    df = rename_with_aliases(
        read_csv_flexible(path),
        {
            "source": ["source"],
            "scenario": ["scenario", "variant"],
            "year": ["year", "anno"],
            "population": ["population", "popolazione", "value", "valore"],
        },
    )
    required = {"source", "scenario", "year", "population"}
    if required - set(df.columns):
        return pd.DataFrame(columns=["source", "scenario", "year", "population"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df = df.dropna(subset=["year", "population"]).copy()
    if df["population"].abs().median() < 100_000:
        df["population"] = df["population"] * 1000.0
    df["year"] = df["year"].astype(int)
    return df[["source", "scenario", "year", "population"]]


def clean_measure_label(value: object) -> str:
    """Normalize benchmark measure labels to lower/median/upper variants."""

    text = normalize_label(value)
    text = text.replace(" ", "_")
    text = text.replace("%", "")
    mapping = {
        "median": "median",
        "medium": "median",
        "central": "median",
        "mean": "median",
        "lower90": "lower90",
        "lower_90": "lower90",
        "lower_at_90": "lower90",
        "lower_90": "lower90",
        "upper90": "upper90",
        "higher90": "upper90",
        "higher_at_90": "upper90",
        "upper_90": "upper90",
        "upper_90": "upper90",
        "lower95": "lower95",
        "lower_95": "lower95",
        "lower_95": "lower95",
        "upper95": "upper95",
        "upper_95": "upper95",
        "upper_95": "upper95",
    }
    return mapping.get(text, text)


def load_tfr_benchmarks(data_dir: Path) -> pd.DataFrame:
    """Load official TFR benchmark trajectories when available.

    Accepted files:
        tfr_benchmarks.csv
        tfr_altri.xlsx
        tfr altri.xlsx

    The Excel format used in the component plotting script is also supported:
    first column = source, second column = statistic/measure, following columns
    are years.
    """

    candidates = [
        data_dir / "tfr_benchmarks.csv",
        data_dir / "tfr_altri.xlsx",
        data_dir / "tfr altri.xlsx",
        data_dir.parent / "tfr_altri.xlsx",
        data_dir.parent / "tfr altri.xlsx",
        Path("/Users/andreaballerini/Downloads/tfr altri.xlsx"),
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return pd.DataFrame(columns=["source", "measure", "year", "tfr"])

    if path.suffix.lower() == ".csv":
        df = rename_with_aliases(
            read_csv_flexible(path),
            {
                "source": ["source"],
                "measure": ["measure", "stat", "scenario", "variant"],
                "year": ["year", "anno"],
                "tfr": ["tfr", "value", "valore"],
            },
        )
        if {"source", "measure", "year", "tfr"} - set(df.columns):
            return pd.DataFrame(columns=["source", "measure", "year", "tfr"])
    else:
        raw = pd.read_excel(path)
        if raw.shape[1] < 3:
            return pd.DataFrame(columns=["source", "measure", "year", "tfr"])
        raw = raw.rename(columns={raw.columns[0]: "source", raw.columns[1]: "measure"})
        raw["source"] = raw["source"].ffill()
        year_cols = [c for c in raw.columns if str(c).isdigit()]
        if not year_cols:
            return pd.DataFrame(columns=["source", "measure", "year", "tfr"])
        df = raw[["source", "measure"] + year_cols].melt(
            id_vars=["source", "measure"],
            value_vars=year_cols,
            var_name="year",
            value_name="tfr",
        )

    df["source"] = df["source"].map(standardize_source)
    df["measure"] = df["measure"].map(clean_measure_label)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["tfr"] = pd.to_numeric(df["tfr"], errors="coerce")
    df = df.dropna(subset=["source", "measure", "year", "tfr"]).copy()
    df["year"] = df["year"].astype(int)
    return df[["source", "measure", "year", "tfr"]].sort_values(["source", "measure", "year"])


def load_e0_benchmarks(data_dir: Path) -> pd.DataFrame:
    """Load official e0 benchmark trajectories when available."""

    candidates = [
        data_dir / "e0_benchmarks.csv",
        data_dir / "mortality_benchmarks_e0.csv",
        data_dir / "e0altri.xlsx",
        data_dir.parent / "e0altri.xlsx",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return pd.DataFrame(columns=["source", "sex", "measure", "year", "e0"])

    if path.suffix.lower() == ".csv":
        df = rename_with_aliases(
            read_csv_flexible(path),
            {
                "source": ["source", "producer", "ente", "benchmark"],
                "sex": ["sex", "sesso", "gender"],
                "measure": ["measure", "scenario", "variant", "type"],
                "year": ["year", "anno"],
                "e0": ["e0", "value", "valore", "life_expectancy", "life_expectancy_at_birth"],
            },
        )
    else:
        df = rename_with_aliases(
            pd.read_excel(path),
            {
                "source": ["source", "producer", "ente", "benchmark"],
                "sex": ["sex", "sesso", "gender"],
                "measure": ["measure", "scenario", "variant", "type"],
                "year": ["year", "anno"],
                "e0": ["value", "e0", "life_expectancy", "life_expectancy_at_birth"],
            },
        )
    if {"source", "sex", "measure", "year", "e0"} - set(df.columns):
        return pd.DataFrame(columns=["source", "sex", "measure", "year", "e0"])
    df["source"] = df["source"].map(standardize_source)
    df["sex"] = df["sex"].map(standardize_sex)
    df["measure"] = df["measure"].map(clean_measure_label)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["e0"] = pd.to_numeric(df["e0"], errors="coerce")
    df = df[df["sex"].isin(SEXES)].dropna(subset=["source", "measure", "year", "e0"]).copy()
    df["year"] = df["year"].astype(int)
    return df[["source", "sex", "measure", "year", "e0"]].sort_values(["source", "sex", "measure", "year"])


# =============================================================================
# Plotting helpers
# =============================================================================

def add_policy_span(ax: plt.Axes, policy_start_year: int, policy_end_year: int) -> None:
    """Highlight years where an immigration policy is active."""

    if policy_end_year >= policy_start_year:
        ax.axvspan(policy_start_year, policy_end_year + 1, color="#F58518", alpha=0.10, label="Policy years")


def plot_total_population(
    summary: pd.DataFrame,
    scenario_names: list[str],
    policy_start_year: int,
    policy_end_year: int,
    official_benchmarks: pd.DataFrame | None = None,
) -> plt.Figure:
    """Plot total population with uncertainty bands."""

    fig, ax = plt.subplots(figsize=LARGE_FIGSIZE)
    add_policy_span(ax, policy_start_year, policy_end_year)
    for _, g in summary[summary["scenario"].isin(scenario_names)].groupby("scenario"):
        g = g.sort_values("year")
        ax.plot(g["year"], g["p50"], linewidth=2.4, label=g["scenario_label"].iloc[0])
        ax.fill_between(g["year"].to_numpy(float), g["p10"].to_numpy(float), g["p90"].to_numpy(float), alpha=0.14)
    if official_benchmarks is not None and not official_benchmarks.empty:
        for (source, scenario), g in official_benchmarks.groupby(["source", "scenario"]):
            g = g.sort_values("year")
            ax.plot(g["year"], g["population"], linestyle="--", linewidth=1.6, alpha=0.78, label=f"{source}: {scenario}")
    ax.set_title("Projected total population")
    ax.set_xlabel("Year")
    ax.set_ylabel("Persons")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_summary_lines(
    summary: pd.DataFrame,
    scenario_names: list[str],
    title: str,
    ylabel: str,
    policy_start_year: int,
    policy_end_year: int,
) -> plt.Figure:
    """Plot p50 lines for a projected indicator."""

    fig, ax = plt.subplots(figsize=LARGE_FIGSIZE)
    add_policy_span(ax, policy_start_year, policy_end_year)
    for _, g in summary[summary["scenario"].isin(scenario_names)].groupby("scenario"):
        g = g.sort_values("year")
        ax.plot(g["year"], g["p50"], linewidth=2.2, label=g["scenario_label"].iloc[0])
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_total_tfr(
    total_tfr_summary: pd.DataFrame,
    scenario_names: list[str],
    tfr_benchmarks: pd.DataFrame,
    policy_start_year: int,
    policy_end_year: int,
) -> plt.Figure:
    """Plot total-population TFR with official benchmark lines."""

    fig, ax = plt.subplots(figsize=LARGE_FIGSIZE)
    add_policy_span(ax, policy_start_year, policy_end_year)
    for _, g in total_tfr_summary[total_tfr_summary["scenario"].isin(scenario_names)].groupby("scenario"):
        g = g.sort_values("year")
        ax.plot(g["year"], g["p50"], linewidth=2.2, label=g["scenario_label"].iloc[0])
        ax.fill_between(g["year"].to_numpy(float), g["p10"].to_numpy(float), g["p90"].to_numpy(float), alpha=0.12)

    if not tfr_benchmarks.empty:
        for source, b in tfr_benchmarks.groupby("source"):
            color = SOURCE_COLORS.get(source, None)
            median = b[b["measure"].eq("median")].sort_values("year")
            if not median.empty:
                ax.plot(
                    median["year"],
                    median["tfr"],
                    linestyle="--",
                    linewidth=2.4,
                    alpha=0.95,
                    color=color,
                    label=f"{source} median",
                )
            lower = b[b["measure"].isin(["lower90", "lower95"])].sort_values("year")
            upper = b[b["measure"].isin(["upper90", "upper95"])].sort_values("year")
            if not lower.empty and not upper.empty:
                band = lower[["year", "tfr"]].merge(upper[["year", "tfr"]], on="year", suffixes=("_lower", "_upper"))
                ax.fill_between(
                    band["year"].to_numpy(float),
                    band["tfr_lower"].to_numpy(float),
                    band["tfr_upper"].to_numpy(float),
                    color=color,
                    alpha=0.16,
                    label=f"{source} interval",
                )

    ax.set_title("Total TFR: projection and official comparisons")
    ax.set_xlabel("Year")
    ax.set_ylabel("TFR")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    return fig


def plot_tfr_by_citizenship(tfr_summary: pd.DataFrame, policy_start_year: int, policy_end_year: int) -> plt.Figure:
    """Plot forecast TFR by citizenship for diagnostic detail."""

    fig, ax = plt.subplots(figsize=LARGE_FIGSIZE)
    add_policy_span(ax, policy_start_year, policy_end_year)
    for citizenship, g in tfr_summary.groupby("citizenship"):
        g = g.sort_values("year")
        ax.plot(g["year"], g["tfr_p50"], linewidth=2.2, label=citizenship)
        ax.fill_between(g["year"].to_numpy(float), g["tfr_p10"].to_numpy(float), g["tfr_p90"].to_numpy(float), alpha=0.14)
    ax.set_title("Forecast TFR by citizenship")
    ax.set_xlabel("Year")
    ax.set_ylabel("TFR")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_e0(
    e0_summary: pd.DataFrame,
    e0_benchmarks: pd.DataFrame,
    policy_start_year: int,
    policy_end_year: int,
) -> plt.Figure:
    """Plot forecast life expectancy at birth by sex with benchmark lines."""

    fig, ax = plt.subplots(figsize=LARGE_FIGSIZE)
    add_policy_span(ax, policy_start_year, policy_end_year)
    colors = OUR_COLORS
    for sex, g in e0_summary.groupby("sex"):
        g = g.sort_values("year")
        ax.plot(g["year"], g["e0_p50"], linewidth=2.3, color=colors.get(sex), label=f"Our {sex}")
        ax.fill_between(g["year"].to_numpy(float), g["e0_p10"].to_numpy(float), g["e0_p90"].to_numpy(float), color=colors.get(sex), alpha=0.13)

    if not e0_benchmarks.empty:
        sex_styles = {"male": "--", "female": ":"}
        for (source, sex), b in e0_benchmarks.groupby(["source", "sex"]):
            color = SOURCE_COLORS.get(source, None)
            median = b[b["measure"].eq("median")].sort_values("year")
            if not median.empty:
                ax.plot(
                    median["year"],
                    median["e0"],
                    linestyle=sex_styles.get(sex, "--"),
                    linewidth=2.1,
                    alpha=0.95,
                    color=color,
                    label=f"{source} {sex} median",
                )
            lower = b[b["measure"].isin(["lower90", "lower95"])].sort_values("year")
            upper = b[b["measure"].isin(["upper90", "upper95"])].sort_values("year")
            if not lower.empty and not upper.empty:
                band = lower[["year", "e0"]].merge(upper[["year", "e0"]], on="year", suffixes=("_lower", "_upper"))
                ax.fill_between(
                    band["year"].to_numpy(float),
                    band["e0_lower"].to_numpy(float),
                    band["e0_upper"].to_numpy(float),
                    color=color,
                    alpha=0.10,
                    label=f"{source} {sex} interval",
                )
    ax.set_title("Life expectancy at birth: projection and official comparisons")
    ax.set_xlabel("Year")
    ax.set_ylabel("e0")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    return fig


def plot_pyramid(pop_mean: pd.DataFrame, scenario_name: str, year: int) -> plt.Figure:
    """Plot a population pyramid for one scenario and year."""

    g = pop_mean[(pop_mean["scenario"].eq(scenario_name)) & (pop_mean["year"].eq(year))].copy()
    wide = g.pivot(index="age", columns="sex", values="population").reindex(AGES).fillna(0.0)
    if "male" not in wide.columns:
        wide["male"] = 0.0
    if "female" not in wide.columns:
        wide["female"] = 0.0

    fig, ax = plt.subplots(figsize=PYRAMID_FIGSIZE)
    ax.barh(wide.index, -wide["male"], color="#4C78A8", alpha=0.76, label="Males")
    ax.barh(wide.index, wide["female"], color="#E45756", alpha=0.76, label="Females")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"Population pyramid, {year}")
    ax.set_xlabel("Population")
    ax.set_ylabel("Age")
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.20)
    fig.tight_layout()
    return fig


def plot_migration_with_benchmark(
    migration_summary: pd.DataFrame,
    benchmarks: pd.DataFrame,
    scenario_names: list[str],
    policy_start_year: int,
    policy_end_year: int,
) -> plt.Figure:
    """Plot projected net migration and optional external benchmark medians."""

    fig, ax = plt.subplots(figsize=LARGE_FIGSIZE)
    add_policy_span(ax, policy_start_year, policy_end_year)
    gsum = migration_summary[
        migration_summary["scenario"].isin(scenario_names)
        & migration_summary["indicator"].eq("net_migration")
    ].copy()
    for _, g in gsum.groupby("scenario"):
        g = g.sort_values("year")
        ax.plot(g["year"], g["p50"], linewidth=2.2, label=g["scenario_label"].iloc[0])
        ax.fill_between(g["year"].to_numpy(float), g["p10"].to_numpy(float), g["p90"].to_numpy(float), alpha=0.11)

    if not benchmarks.empty:
        b = benchmarks[
            benchmarks["flow"].astype(str).str.lower().str.contains("net")
            & benchmarks["measure"].astype(str).str.lower().str.contains("median")
        ].copy()
        for source, gg in b.groupby("source"):
            gg = gg.sort_values("year")
            ax.plot(gg["year"], gg["value"], linestyle="--", linewidth=1.5, alpha=0.75, label=f"{source} median")

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.55)
    ax.set_title("Net migration: projection scenarios and benchmarks")
    ax.set_xlabel("Year")
    ax.set_ylabel("Persons")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


# =============================================================================
# Reverse-policy search
# =============================================================================

def build_central_components(
    data_dir: Path,
    start_year: int,
    end_year: int,
    mortality_version: str,
    tfr_increase_pct: float,
) -> dict:
    """Build central component paths once for fast deterministic policy search."""

    rng = np.random.default_rng(RANDOM_SEED)
    transition_years = np.arange(start_year, end_year, dtype=int)
    state_years = np.arange(start_year, end_year + 1, dtype=int)
    initial_state, _, initial_population_year = load_initial_population(data_dir, start_year)
    fertility = build_fertility_forecast(data_dir, transition_years, 1, rng, tfr_increase_pct)
    mortality = build_mortality_forecast(data_dir, transition_years, 1, rng, mortality_version)
    migration = build_migration_forecast(data_dir, transition_years, 1, rng, initial_state)
    return {
        "transition_years": transition_years,
        "state_years": state_years,
        "initial_state": initial_state,
        "initial_population_year": initial_population_year,
        "fertility": fertility,
        "mortality": mortality,
        "migration": migration,
    }


def deterministic_population_path(components: dict, scenario: Scenario) -> pd.DataFrame:
    """Run one central deterministic projection path for reverse searches."""

    state = components["initial_state"].copy()
    rows = [{"year": int(components["state_years"][0]), "population": float(state.sum())}]
    for y_idx, year in enumerate(components["transition_years"]):
        px = components["mortality"]["px_central"][y_idx, :, :]
        asfr = components["fertility"]["asfr_central"][y_idx, :, :]
        base_immigration = components["migration"]["single_age_mean"][FLOW_TO_INDEX["immigration"], y_idx, :, :, :]
        base_emigration = components["migration"]["single_age_mean"][FLOW_TO_INDEX["emigration"], y_idx, :, :, :]
        immigration, emigration = apply_migration_scenario(
            base_immigration, base_emigration, scenario, int(year), components["initial_state"]
        )
        state, _, _ = project_one_year(state, px, asfr, immigration, emigration)
        rows.append({"year": int(year + 1), "population": float(state.sum())})
    return pd.DataFrame(rows)


def population_at_target(components: dict, scenario: Scenario, target_year: int) -> float:
    """Return central projected population in the selected target year."""

    path = deterministic_population_path(components, scenario)
    row = path[path["year"].eq(int(target_year))]
    if row.empty:
        raise ValueError("Target year is outside the projection horizon.")
    return float(row["population"].iloc[0])


def find_required_extra_immigrants(
    data_dir: Path,
    start_year: int,
    end_year: int,
    mortality_version: str,
    tfr_increase_pct: float,
    target_population: float,
    target_year: int,
    policy_start_year: int,
    policy_end_year: int,
    child_share_of_extra: float,
    adult_min_age: int,
    adult_max_age: int,
    upper_limit: float = 2_000_000.0,
) -> pd.DataFrame:
    """Find annual extra immigrants needed to hit a target population.

    The calculation uses the central component paths, not stochastic intervals.
    This keeps the reverse tool fast enough for Streamlit interaction and makes
    the result easy to interpret: "about X additional immigrants per policy year
    under this deterministic baseline."
    """

    components = build_central_components(data_dir, start_year, end_year, mortality_version, tfr_increase_pct)
    policy_scenarios = [
        Scenario(
            name="higher_total_immigration",
            label="Policy: higher total immigration only",
            policy_start_year=policy_start_year,
            policy_end_year=policy_end_year,
            adult_min_age=adult_min_age,
            adult_max_age=adult_max_age,
        ),
        Scenario(
            name="higher_immigration_younger_with_children",
            label="Policy: higher immigration + younger age structure + children",
            policy_start_year=policy_start_year,
            policy_end_year=policy_end_year,
            younger_with_children=True,
            child_share_of_extra=child_share_of_extra,
            adult_min_age=adult_min_age,
            adult_max_age=adult_max_age,
        ),
    ]

    rows = []
    for scenario in policy_scenarios:
        base_value = population_at_target(components, scenario, target_year)
        if base_value >= target_population:
            rows.append(
                {
                    "scenario": scenario.label,
                    "target_year": int(target_year),
                    "target_population": float(target_population),
                    "required_additional_immigrants_per_year": 0.0,
                    "achieved_population": base_value,
                    "gap_after_solution": base_value - target_population,
                    "policy_years": int(policy_end_year - policy_start_year + 1),
                    "status": "Target already reached without extra immigrants",
                }
            )
            continue

        lo, hi = 0.0, 100_000.0
        while hi < upper_limit:
            trial = Scenario(
                **{
                    **scenario.__dict__,
                    "extra_immigration_per_year": hi,
                }
            )
            value = population_at_target(components, trial, target_year)
            if value >= target_population:
                break
            lo = hi
            hi *= 2.0

        if hi >= upper_limit:
            trial = Scenario(**{**scenario.__dict__, "extra_immigration_per_year": upper_limit})
            value = population_at_target(components, trial, target_year)
            rows.append(
                {
                    "scenario": scenario.label,
                    "target_year": int(target_year),
                    "target_population": float(target_population),
                    "required_additional_immigrants_per_year": np.nan,
                    "achieved_population": value,
                    "gap_after_solution": value - target_population,
                    "policy_years": int(policy_end_year - policy_start_year + 1),
                    "status": f"Not reached below {upper_limit:,.0f} extra immigrants/year",
                }
            )
            continue

        for _ in range(28):
            mid = (lo + hi) / 2.0
            trial = Scenario(**{**scenario.__dict__, "extra_immigration_per_year": mid})
            value = population_at_target(components, trial, target_year)
            if value >= target_population:
                hi = mid
            else:
                lo = mid

        solution = hi
        solved = Scenario(**{**scenario.__dict__, "extra_immigration_per_year": solution})
        achieved = population_at_target(components, solved, target_year)
        rows.append(
            {
                "scenario": scenario.label,
                "target_year": int(target_year),
                "target_population": float(target_population),
                "required_additional_immigrants_per_year": float(solution),
                "achieved_population": float(achieved),
                "gap_after_solution": float(achieved - target_population),
                "policy_years": int(policy_end_year - policy_start_year + 1),
                "status": "Solved",
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Cached Streamlit runner
# =============================================================================

@st.cache(show_spinner=True, allow_output_mutation=True)
def run_projection_cached(
    data_dir_str: str,
    start_year: int,
    end_year: int,
    n_sim: int,
    mortality_version: str,
    tfr_increase_pct: float,
    selected_names: tuple[str, ...],
    policy_start_year: int,
    policy_end_year: int,
    extra_immigration_per_year: float,
    child_share_of_extra: float,
    adult_min_age: int,
    adult_max_age: int,
) -> dict:
    """Cached wrapper so interaction does not refit components unnecessarily."""

    data_dir = Path(data_dir_str)
    scenarios_all = make_scenarios(
        policy_start_year=policy_start_year,
        policy_end_year=policy_end_year,
        extra_immigration_per_year=extra_immigration_per_year,
        child_share_of_extra=child_share_of_extra,
        adult_min_age=adult_min_age,
        adult_max_age=adult_max_age,
    )
    scenarios = [s for s in scenarios_all if s.name in set(selected_names)]
    return run_projection(
        data_dir=data_dir,
        start_year=int(start_year),
        end_year=int(end_year),
        n_sim=int(n_sim),
        mortality_version=mortality_version,
        scenarios=scenarios,
        tfr_increase_pct=float(tfr_increase_pct),
    )


@st.cache(show_spinner=True, allow_output_mutation=True)
def reverse_search_cached(
    data_dir_str: str,
    start_year: int,
    end_year: int,
    mortality_version: str,
    tfr_increase_pct: float,
    target_population: float,
    target_year: int,
    policy_start_year: int,
    policy_end_year: int,
    child_share_of_extra: float,
    adult_min_age: int,
    adult_max_age: int,
) -> pd.DataFrame:
    """Cached reverse-search wrapper."""

    return find_required_extra_immigrants(
        data_dir=Path(data_dir_str),
        start_year=int(start_year),
        end_year=int(end_year),
        mortality_version=mortality_version,
        tfr_increase_pct=float(tfr_increase_pct),
        target_population=float(target_population),
        target_year=int(target_year),
        policy_start_year=int(policy_start_year),
        policy_end_year=int(policy_end_year),
        child_share_of_extra=float(child_share_of_extra),
        adult_min_age=int(adult_min_age),
        adult_max_age=int(adult_max_age),
    )


# =============================================================================
# Macro simulation helpers (cached)
# =============================================================================

@st.cache(show_spinner=False, allow_output_mutation=True)
def _cached_load_macro_data(rdata_path: str):
    """Load df_proj and model data from macro/ CSVs. Returns (df_proj, model_data, error_str)."""
    if not _MACRO_AVAILABLE:
        return None, None, "macro_simulations module not available."
    try:
        df_proj    = load_macro_data(rdata_path)
        model_data = load_model_data(rdata_path)
        return df_proj, model_data, None
    except Exception as exc:
        return None, None, str(exc)


# =============================================================================
# Streamlit interface
# =============================================================================

def main() -> None:
    """Build the Streamlit app."""

    st.set_page_config(page_title="Italy population projection", layout="wide")
    st.title("Italy cohort-component population projection")
    st.caption(
        "Self-contained version: raw fertility, mortality, migration, and population files are read directly "
        "from the GitHub repository. No precomputed output tables and no external projection engine are used."
    )

    scenario_options = {
        "baseline_migration": "Baseline migration scenario",
        "no_immigration": "No immigration scenario",
        "higher_total_immigration": "Higher total immigration only",
        "higher_immigration_younger_with_children": "Higher immigration + younger age structure + children",
    }

    with st.sidebar:
        st.header("Input folder")
        data_dir = Path(st.text_input("Repository/input folder", value=str(APP_DIR))).expanduser()

        st.header("Projection settings")
        start_year = st.number_input("Start year", min_value=2000, max_value=2100, value=2025, step=1)
        end_year = st.number_input("End year", min_value=int(start_year) + 1, max_value=2150, value=2075, step=1)
        n_sim = st.slider("Simulation paths", min_value=10, max_value=200, value=40, step=10)
        mortality_version = st.selectbox("Mortality version", ["excluding_covid", "all_years"], index=0)
        tfr_increase_pct = st.slider("Increase forecast TFR by", 0.0, 50.0, 0.0, 1.0, format="%.0f%%")

        st.header("Scenarios")
        selected_names = st.multiselect(
            "Scenarios to display",
            options=list(scenario_options.keys()),
            default=list(scenario_options.keys()),
            format_func=lambda x: scenario_options[x],
        )

        st.header("Policy parameters")
        policy_start_year = st.slider("Policy start year", int(start_year), int(end_year) - 1, max(int(start_year), 2026))
        policy_end_year = st.slider("Policy end year", policy_start_year, int(end_year) - 1, int(end_year) - 1)
        extra_immigration_per_year = st.number_input(
            "Additional immigrants per year", min_value=0, max_value=500_000, value=0, step=25_000
        )
        child_share_of_extra = st.slider("Child share of extra immigrants in scenario 3", 0.0, 0.60, 0.25, 0.05)
        adult_min_age = st.number_input("Policy adult minimum age", min_value=0, max_value=100, value=20, step=1)
        adult_max_age = st.number_input("Policy adult maximum age", min_value=int(adult_min_age), max_value=100, value=max(34, int(adult_min_age)), step=1)

    if not selected_names:
        st.warning("Select at least one scenario.")
        st.stop()

    try:
        results = run_projection_cached(
            str(data_dir),
            int(start_year),
            int(end_year),
            int(n_sim),
            mortality_version,
            float(tfr_increase_pct),
            tuple(selected_names),
            int(policy_start_year),
            int(policy_end_year),
            float(extra_immigration_per_year),
            float(child_share_of_extra),
            int(adult_min_age),
            int(adult_max_age),
        )
    except Exception as exc:
        st.error(f"Projection could not run: {exc}")
        st.info(
            "Check that the repository contains population.csv or italiani.csv + stranieri.csv, "
            "asfr.csv, TFR.csv, mortality.csv, immigrati_input.csv, and emigrati_input.csv."
        )
        st.stop()

    # Load macro data (cached per rdata path)
    _rdata_path = str(data_dir / "estimations.RData")
    _df_proj_macro, _model_data, _macro_err = _cached_load_macro_data(_rdata_path)

    total = results["population_total_summary"]
    births = results["births_summary"]
    oadr = results["old_age_dependency_summary"]
    migration = results["migration_summary"]
    total_tfr = results["total_tfr_summary"]
    pop_mean = results["population_by_age_mean"]
    benchmarks = load_migration_benchmarks(data_dir)
    official_population_benchmarks = load_official_population_benchmarks(data_dir)
    tfr_benchmarks = load_tfr_benchmarks(data_dir)
    e0_benchmarks = load_e0_benchmarks(data_dir)

    final_year = int(total["year"].max())
    base_final = total[(total["scenario"].eq("baseline_migration")) & (total["year"].eq(final_year))]

    c1, c2, c3, c4 = st.columns(4)
    if len(base_final):
        c1.metric(f"Baseline population {final_year}", f"{base_final['p50'].iloc[0]:,.0f}")
    c2.metric("Simulation paths", f"{n_sim}")
    c3.metric("Initial population year", str(results["initial_population_year"]))
    c4.metric("Mortality version", mortality_version.replace("_", " "))

    st.subheader("Projected total population")
    st.pyplot(
        plot_total_population(
            total,
            list(selected_names),
            int(policy_start_year),
            int(policy_end_year),
            official_population_benchmarks,
        ),
        clear_figure=True,
    )
    if official_population_benchmarks.empty:
        st.caption(
            "Official projection comparison is ready in the code, but no "
            "official_population_projection_comparisons.csv file is present yet."
        )

    st.subheader("Projected births")
    st.pyplot(
        plot_summary_lines(
            births,
            list(selected_names),
            "Projected births",
            "Births",
            int(policy_start_year),
            int(policy_end_year),
        ),
        clear_figure=True,
    )

    st.subheader("Old-age dependency ratio")
    st.pyplot(
        plot_summary_lines(
            oadr,
            list(selected_names),
            "Old-age dependency ratio",
            "Population 65+ / population 15-64 (%)",
            int(policy_start_year),
            int(policy_end_year),
        ),
        clear_figure=True,
    )

    st.subheader("Net migration")
    st.pyplot(
        plot_migration_with_benchmark(
            migration,
            benchmarks,
            list(selected_names),
            int(policy_start_year),
            int(policy_end_year),
        ),
        clear_figure=True,
    )

    st.subheader("Total TFR")
    st.pyplot(
        plot_total_tfr(
            total_tfr,
            list(selected_names),
            tfr_benchmarks,
            int(policy_start_year),
            int(policy_end_year),
        ),
        clear_figure=True,
    )
    if tfr_benchmarks.empty:
        st.caption("No TFR benchmark file found. Add tfr_benchmarks.csv, tfr_altri.xlsx, or tfr altri.xlsx to the repo root.")

    st.subheader("Forecast life expectancy")
    st.pyplot(plot_e0(results["e0_summary"], e0_benchmarks, int(policy_start_year), int(policy_end_year)), clear_figure=True)
    if e0_benchmarks.empty:
        st.caption("No e0 benchmark file found. Add e0altri.xlsx or e0_benchmarks.csv to the repo root.")

    st.subheader("Population pyramid")
    pyramid_year = st.slider("Pyramid year", int(start_year), int(end_year), int(end_year))
    pyramid_scenario = st.selectbox(
        "Pyramid scenario",
        options=list(selected_names),
        format_func=lambda x: scenario_options[x],
    )
    st.pyplot(plot_pyramid(pop_mean, pyramid_scenario, int(pyramid_year)), clear_figure=True)

    st.subheader("Reverse policy calculator")
    st.caption(
        "Central deterministic calculation: choose a target population and target year; "
        "the app estimates the annual additional immigrants needed during the selected policy years."
    )
    reverse_cols = st.columns(3)
    with reverse_cols[0]:
        target_year = st.number_input("Target year", min_value=int(start_year) + 1, max_value=int(end_year), value=int(end_year), step=1)
    with reverse_cols[1]:
        default_target = float(total[total["year"].eq(int(target_year))]["p50"].max()) if len(total[total["year"].eq(int(target_year))]) else 50_000_000.0
        target_population = st.number_input(
            "Desired population",
            min_value=0,
            max_value=100_000_000,
            value=int(round(default_target / 1_000_000) * 1_000_000),
            step=500_000,
        )
    with reverse_cols[2]:
        run_reverse = st.button("Compute required immigration")

    if run_reverse:
        reverse = reverse_search_cached(
            str(data_dir),
            int(start_year),
            int(end_year),
            mortality_version,
            float(tfr_increase_pct),
            float(target_population),
            int(target_year),
            int(policy_start_year),
            int(policy_end_year),
            float(child_share_of_extra),
            int(adult_min_age),
            int(adult_max_age),
        )
        st.dataframe(
            reverse.style.format(
                {
                    "target_population": "{:,.0f}",
                    "required_additional_immigrants_per_year": "{:,.0f}",
                    "achieved_population": "{:,.0f}",
                    "gap_after_solution": "{:,.0f}",
                }
            ),
            use_container_width=True,
        )

    with st.expander("Intermediate demographic indexes"):
        st.write("Total TFR forecast")
        st.dataframe(total_tfr, use_container_width=True)
        st.write("TFR forecast")
        st.dataframe(results["tfr_summary"], use_container_width=True)
        st.pyplot(plot_tfr_by_citizenship(results["tfr_summary"], int(policy_start_year), int(policy_end_year)), clear_figure=True)
        st.write("Life expectancy at birth forecast")
        st.dataframe(results["e0_summary"], use_container_width=True)

    with st.expander("Download result tables"):
        st.download_button(
            "Download total population summary",
            total.to_csv(index=False).encode("utf-8"),
            file_name="population_total_summary.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download population by age mean",
            pop_mean.to_csv(index=False).encode("utf-8"),
            file_name="population_by_age_mean.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download migration summary",
            migration.to_csv(index=False).encode("utf-8"),
            file_name="migration_summary.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download total TFR summary",
            total_tfr.to_csv(index=False).encode("utf-8"),
            file_name="total_tfr_summary.csv",
            mime="text/csv",
        )

    # -------------------------------------------------------------------------
    # Macroeconomic simulations expander
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown(
        "<h2 style='margin-bottom:0px'>Macroeconomic Projections for Italy</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#555;font-size:0.9em;margin-top:4px'>"
        "Macroeconomic projections with an conometric model linking Italian demographics to GDP growth, "
        "inflation, interest rates, and fiscal dynamics."
        "</p>",
        unsafe_allow_html=True,
    )
    with st.expander("Show / hide macro simulation plots", expanded=True):
        if not _MACRO_AVAILABLE:
            st.warning("macro_simulations.py could not be imported. Ensure it is in the same directory.")
        elif _macro_err:
            st.warning(
                f"Could not load macro data: {_macro_err}\n\n"
                "Place **estimations.RData** in the input folder and ensure **pyreadr** is installed."
            )
        else:
            _MACRO_N_SIM = 300

            _macro_col1, _macro_col2 = st.columns(2)
            with _macro_col1:
                _end_year_macro = st.slider(
                    "Projection end year", 2030, 2075, 2050, 5, key="macro_end_year"
                )
            with _macro_col2:
                _start_year_macro = st.slider(
                    "Historical data starts from", 1970, 2024, 2010, 5, key="macro_start_year"
                )

            # ------------------------------------------------------------------
            # Retirement reform settings
            # ------------------------------------------------------------------
            _enable_reforms = st.checkbox(
                "Enable retirement age reforms",
                value=False,
                key="macro_enable_reforms",
                help=(
                    "Simulate the effect of raising the statutory retirement age. "
                    "People who would have retired stay in the working-age group (N_3), "
                    "shrinking the retired group (N_4) and revising demographic proxies "
                    "such as D_R, MY, and YM used in the macro model."
                ),
            )

            _retirement_schedule: dict | None = None
            if _enable_reforms:
                st.markdown(
                    "<p style='color:#555;font-size:0.88em;margin-bottom:6px'>"
                    "Current baseline retirement age is <b>65</b>. "
                    "Reforms take effect at the specified year and persist until the next step.</p>",
                    unsafe_allow_html=True,
                )
                _n_reforms = int(
                    st.number_input(
                        "Number of reform steps",
                        min_value=1, max_value=10, value=3, step=1,
                        key="macro_n_reforms",
                    )
                )
                _rh_col1, _rh_col2 = st.columns(2)
                _rh_col1.markdown("**Reform year**")
                _rh_col2.markdown("**Retirement age**")
                _retirement_schedule = {}
                for _ri in range(_n_reforms):
                    _rc1, _rc2 = st.columns(2)
                    _reform_yr = int(
                        _rc1.number_input(
                            f"Year {_ri + 1}",
                            min_value=2025, max_value=2074,
                            value=min(2025 + _ri * 5, 2074),
                            step=1,
                            key=f"macro_reform_year_{_ri}",
                            label_visibility="collapsed",
                        )
                    )
                    _reform_age = int(
                        _rc2.number_input(
                            f"Retirement age {_ri + 1}",
                            min_value=60, max_value=85,
                            value=min(67 + _ri * 2, 85),
                            step=1,
                            key=f"macro_reform_age_{_ri}",
                            label_visibility="collapsed",
                        )
                    )
                    _retirement_schedule[_reform_yr] = _reform_age

            if st.button("Run Macroeconomic Simulations"):
                with st.spinner("Running 6-country simulation (~10–30 s)…"):
                    try:
                        _italy_demo = compute_demo_ratios_from_projection(
                            results["population_by_age_mean"],
                            retirement_schedule=_retirement_schedule,
                        )
                        _macro_results = run_macro_simulations(
                            _df_proj_macro,
                            _italy_demo,
                            _model_data,
                            num_sim=_MACRO_N_SIM,
                        )
                        st.session_state["macro_results"] = _macro_results
                        st.session_state["macro_df_proj"] = _df_proj_macro
                        st.session_state["macro_italy_demo"] = _italy_demo
                    except Exception as _exc:
                        st.error(f"Simulation failed: {_exc}")

            if "macro_results" in st.session_state:
                _mr = st.session_state["macro_results"]
                _dp = st.session_state.get("macro_df_proj", _df_proj_macro)

                # Download button — flatten {scenario: {var: df}} into one CSV
                _dl_frames = []
                for _sc, _vars in _mr.items():
                    for _v, _df in _vars.items():
                        _tmp = _df.copy()
                        _tmp.insert(0, "scenario", _sc)
                        _tmp.insert(1, "variable", _v)
                        _dl_frames.append(_tmp)
                if _dl_frames:
                    _dl_csv = pd.concat(_dl_frames, ignore_index=True).to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download macro simulation data (CSV)",
                        _dl_csv,
                        file_name="macro_simulations_italy.csv",
                        mime="text/csv",
                    )

                for _var, _label in VAR_LABELS.items():
                    st.subheader(_label)
                    _fig = plot_macro_italy(
                        _mr, _dp, _var,
                        start_year=_start_year_macro,
                        end_year=_end_year_macro,
                    )
                    st.pyplot(_fig, clear_figure=True)

                if "macro_italy_demo" in st.session_state:
                    st.subheader("Old-Age Dependency Ratio (OADR)")
                    _fig_oadr = plot_oadr_italy(
                        st.session_state["macro_italy_demo"],
                        _dp,
                        start_year=_start_year_macro,
                        end_year=_end_year_macro,
                    )
                    st.pyplot(_fig_oadr, clear_figure=True)


if __name__ == "__main__":
    main()
