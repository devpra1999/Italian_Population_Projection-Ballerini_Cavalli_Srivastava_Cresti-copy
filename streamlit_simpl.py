"""
mix_streamlit.py

Streamlit population projection app with the visual/interactive content of
`streamlit_projection_app copy.py`, but with the projection engine written in the
explicit style of `streamlit_app_b copy.py`.

The key difference is the cohort-component step:
    1. keep a dataframe with one row per age;
    2. multiply each age/sex/citizenship population by survival probabilities;
    3. subtract emigration and add immigration;
    4. compute births from female exposure and fertility rates;
    5. age everyone forward by one year;
    6. insert newborns at age 0.

This file intentionally does not edit any existing app file.  It imports the raw
input readers, component forecast builders, benchmark loaders, and plot helpers
from `streamlit_projection_app copy.py`, then swaps only the projection engine
for the more transparent dataframe-based version.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(APP_DIR / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(APP_DIR / ".cache"))

import matplotlib.pyplot as plt
import streamlit as st


SOURCE_APP_PATH = APP_DIR / "final_streamlit.py"


@st.cache_resource(show_spinner=False)
def load_source_app():
    """Load helper functions from the existing copy app without editing it."""

    module_name = "streamlit_projection_app_copy_helpers"
    spec = importlib.util.spec_from_file_location(module_name, SOURCE_APP_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helper app from {SOURCE_APP_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def state_array_to_classic_df(engine, state: np.ndarray) -> pd.DataFrame:
    """Convert citizenship x sex x age array into prototype-style age dataframe."""

    # The component builders inherited from the newer app store population as a
    # 3-dimensional array:
    #     citizenship x sex x age
    #
    # For the projection step we deliberately switch to the easier prototype
    # format: one table, one row per age, one population column for each
    # citizenship-sex group.
    return pd.DataFrame(
        {
            "eta": engine.AGES,
            "maschi_italiani": state[engine.CIT_TO_INDEX["italiani"], engine.SEX_TO_INDEX["male"], :],
            "femmine_italiani": state[engine.CIT_TO_INDEX["italiani"], engine.SEX_TO_INDEX["female"], :],
            "maschi_stranieri": state[engine.CIT_TO_INDEX["stranieri"], engine.SEX_TO_INDEX["male"], :],
            "femmine_stranieri": state[engine.CIT_TO_INDEX["stranieri"], engine.SEX_TO_INDEX["female"], :],
        }
    )


def classic_df_to_state_array(engine, df: pd.DataFrame) -> np.ndarray:
    """Convert the prototype-style age dataframe back to array form."""

    # Some helper functions from the newer app expect arrays.  This small bridge
    # lets us keep the readable dataframe projection while still reusing those
    # helpers for things like total TFR.
    state = np.zeros((len(engine.CITIZENSHIPS), len(engine.SEXES), len(engine.AGES)), dtype=float)
    state[engine.CIT_TO_INDEX["italiani"], engine.SEX_TO_INDEX["male"], :] = df["maschi_italiani"].to_numpy(float)
    state[engine.CIT_TO_INDEX["italiani"], engine.SEX_TO_INDEX["female"], :] = df["femmine_italiani"].to_numpy(float)
    state[engine.CIT_TO_INDEX["stranieri"], engine.SEX_TO_INDEX["male"], :] = df["maschi_stranieri"].to_numpy(float)
    state[engine.CIT_TO_INDEX["stranieri"], engine.SEX_TO_INDEX["female"], :] = df["femmine_stranieri"].to_numpy(float)
    return state


def apply_survival_classic(engine, df: pd.DataFrame, px: np.ndarray) -> pd.DataFrame:
    """Classic step 1: multiply each age by survival probability."""

    # px means the probability of surviving from this year to next year.
    # It is indexed by sex and age, so male columns use male px and female
    # columns use female px.  Citizenship does not change survival here.
    out = df.copy()
    out["maschi_italiani"] *= px[engine.SEX_TO_INDEX["male"], :]
    out["maschi_stranieri"] *= px[engine.SEX_TO_INDEX["male"], :]
    out["femmine_italiani"] *= px[engine.SEX_TO_INDEX["female"], :]
    out["femmine_stranieri"] *= px[engine.SEX_TO_INDEX["female"], :]
    return out


def apply_migration_classic(
    engine,
    df: pd.DataFrame,
    immigration: np.ndarray,
    emigration: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Classic step 2: subtract emigration, add immigration, cap negatives."""

    # Migration is already age-sex-citizenship specific.  For each column:
    #     population after migration =
    #         survived population - emigrants + immigrants
    #
    # We cap emigrants at the number of survivors in that cell, so the model
    # never creates a negative population at any age.
    out = df.copy()
    applied_emigration = np.zeros_like(emigration)
    columns = {
        ("italiani", "male"): "maschi_italiani",
        ("italiani", "female"): "femmine_italiani",
        ("stranieri", "male"): "maschi_stranieri",
        ("stranieri", "female"): "femmine_stranieri",
    }
    for (citizenship, sex), column in columns.items():
        c = engine.CIT_TO_INDEX[citizenship]
        s = engine.SEX_TO_INDEX[sex]
        current = out[column].to_numpy(float)
        emigrants = np.minimum(emigration[c, s, :], current)
        applied_emigration[c, s, :] = emigrants
        out[column] = current - emigrants + immigration[c, s, :]
    return out, applied_emigration


def compute_births_classic(engine, df: pd.DataFrame, asfr: np.ndarray) -> tuple[float, dict[str, float]]:
    """Classic step 3: compute births from female population and ASFR."""

    # ASFR = age-specific fertility rate.  It tells us how many births are
    # expected per woman at each age.  The fertility component already stores it
    # as a rate, not per 1,000, so we multiply directly.
    births_italiani = float((df["femmine_italiani"].to_numpy(float) * asfr[engine.CIT_TO_INDEX["italiani"], :]).sum())
    births_stranieri = float((df["femmine_stranieri"].to_numpy(float) * asfr[engine.CIT_TO_INDEX["stranieri"], :]).sum())
    return births_italiani + births_stranieri, {
        "italiani": births_italiani,
        "stranieri": births_stranieri,
    }


def age_forward_classic(engine, df: pd.DataFrame, births_by_citizenship: dict[str, float]) -> pd.DataFrame:
    """Classic step 4: shift age x to x+1 and insert newborns at age 0."""

    # Ageing rule:
    # - everyone aged 0 becomes age 1 next year;
    # - everyone aged 1 becomes age 2;
    # - ...
    # - everyone aged 99 becomes age 100;
    # - age 100 is open-ended, so people already at 100 remain in the 100 group.
    #
    # After ageing, age 0 is overwritten by newborns.
    next_df = pd.DataFrame({"eta": engine.AGES})
    for column in ["maschi_italiani", "femmine_italiani", "maschi_stranieri", "femmine_stranieri"]:
        values = df[column].to_numpy(float)
        aged = np.zeros_like(values)
        aged[1:] = values[:-1]
        aged[100] += values[100]
        next_df[column] = aged

    next_df.loc[0, "maschi_italiani"] = births_by_citizenship["italiani"] * engine.SEX_RATIO_AT_BIRTH_MALE
    next_df.loc[0, "femmine_italiani"] = births_by_citizenship["italiani"] * (1.0 - engine.SEX_RATIO_AT_BIRTH_MALE)
    next_df.loc[0, "maschi_stranieri"] = births_by_citizenship["stranieri"] * engine.SEX_RATIO_AT_BIRTH_MALE
    next_df.loc[0, "femmine_stranieri"] = births_by_citizenship["stranieri"] * (1.0 - engine.SEX_RATIO_AT_BIRTH_MALE)
    return next_df


def total_population_classic(df: pd.DataFrame) -> float:
    """Total population in the prototype-style dataframe."""

    return float(
        df["maschi_italiani"].sum()
        + df["femmine_italiani"].sum()
        + df["maschi_stranieri"].sum()
        + df["femmine_stranieri"].sum()
    )


def oadr_classic(df: pd.DataFrame) -> float:
    """Old-age dependency ratio from the prototype-style dataframe."""

    pop_cols = ["maschi_italiani", "femmine_italiani", "maschi_stranieri", "femmine_stranieri"]
    older = df.loc[df["eta"].ge(65), pop_cols].sum().sum()
    working = df.loc[df["eta"].between(15, 64), pop_cols].sum().sum()
    return float(100.0 * older / working) if working > 0 else np.nan


def save_classic_state(engine, df: pd.DataFrame, scen_idx: int, sim_idx: int, year_idx: int, total_pop: np.ndarray, oadr: np.ndarray, pop_age_sex: np.ndarray) -> None:
    """Store summary arrays from the prototype dataframe."""

    total_pop[scen_idx, sim_idx, year_idx] = total_population_classic(df)
    oadr[scen_idx, sim_idx, year_idx] = oadr_classic(df)
    pop_age_sex[scen_idx, sim_idx, year_idx, engine.SEX_TO_INDEX["male"], :] = (
        df["maschi_italiani"].to_numpy(float) + df["maschi_stranieri"].to_numpy(float)
    )
    pop_age_sex[scen_idx, sim_idx, year_idx, engine.SEX_TO_INDEX["female"], :] = (
        df["femmine_italiani"].to_numpy(float) + df["femmine_stranieri"].to_numpy(float)
    )


def run_projection_classic(
    engine,
    data_dir: Path,
    start_year: int,
    end_year: int,
    n_sim: int,
    mortality_version: str,
    scenarios: list,
    tfr_increase_pct: float,
) -> dict:
    """Run the whole projection with the explicit dataframe ageing algorithm."""

    # Reproducibility: using a fixed random seed means the uncertainty paths are
    # the same every time we run the app with the same settings.
    rng = np.random.default_rng(engine.RANDOM_SEED)

    # transition_years are years where we move from year t to t+1.
    # state_years are years where we store a population stock.
    transition_years = np.arange(start_year, end_year, dtype=int)
    state_years = np.arange(start_year, end_year + 1, dtype=int)

    # Build the three demographic components from raw input files:
    # - initial population by age, sex and citizenship;
    # - fertility paths and ASFR schedules;
    # - mortality/survival paths;
    # - migration paths by flow, age, sex and citizenship.
    initial_state, initial_population_df, initial_population_year = engine.load_initial_population(data_dir, start_year)
    fertility = engine.build_fertility_forecast(data_dir, transition_years, n_sim, rng, tfr_increase_pct)
    mortality = engine.build_mortality_forecast(data_dir, transition_years, n_sim, rng, mortality_version)
    migration = engine.build_migration_forecast(data_dir, transition_years, n_sim, rng, initial_state)

    total_pop = np.zeros((len(scenarios), n_sim, len(state_years)), dtype=float)
    births = np.zeros((len(scenarios), n_sim, len(transition_years)), dtype=float)
    oadr = np.zeros((len(scenarios), n_sim, len(state_years)), dtype=float)
    total_tfr = np.zeros((len(scenarios), n_sim, len(transition_years)), dtype=float)
    pop_age_sex = np.zeros((len(scenarios), n_sim, len(state_years), len(engine.SEXES), len(engine.AGES)), dtype=float)
    migration_totals = np.zeros((len(scenarios), n_sim, len(transition_years), 3), dtype=float)

    # Main loop.  This is the all-at-once cohort-component projection:
    # for every scenario, for every stochastic simulation path, for every year,
    # update the full age-sex-citizenship population.
    for scen_idx, scenario in enumerate(scenarios):
        for sim_idx in range(n_sim):
            # Start every simulation from exactly the same observed population.
            current_df = state_array_to_classic_df(engine, initial_state)
            save_classic_state(engine, current_df, scen_idx, sim_idx, 0, total_pop, oadr, pop_age_sex)

            for y_idx, year in enumerate(transition_years):
                # Pick the demographic rates/flows for this simulation and year.
                px = mortality["px_sim"][sim_idx, y_idx, :, :]
                asfr = fertility["asfr_sim"][sim_idx, y_idx, :, :]
                base_immigration = migration["single_age_sim"][sim_idx, engine.FLOW_TO_INDEX["immigration"], y_idx, :, :, :]
                base_emigration = migration["single_age_sim"][sim_idx, engine.FLOW_TO_INDEX["emigration"], y_idx, :, :, :]

                # Apply the selected migration scenario.  The baseline forecast is
                # not overwritten; policy scenarios add extra immigration after
                # the baseline migration path has already been generated.
                immigration, emigration = engine.apply_migration_scenario(
                    base_immigration,
                    base_emigration,
                    scenario,
                    int(year),
                    initial_state,
                )

                total_tfr[scen_idx, sim_idx, y_idx] = engine.compute_total_tfr_from_state(
                    classic_df_to_state_array(engine, current_df),
                    asfr,
                )

                # The readable projection steps:
                # 1. survive;
                # 2. migrate;
                # 3. calculate births;
                # 4. age forward and place newborns at age 0.
                survived_df = apply_survival_classic(engine, current_df, px)
                migrated_df, applied_emigration = apply_migration_classic(engine, survived_df, immigration, emigration)
                total_births, births_by_citizenship = compute_births_classic(engine, migrated_df, asfr)
                next_df = age_forward_classic(engine, migrated_df, births_by_citizenship)

                births[scen_idx, sim_idx, y_idx] = total_births
                migration_totals[scen_idx, sim_idx, y_idx, 0] = immigration.sum()
                migration_totals[scen_idx, sim_idx, y_idx, 1] = applied_emigration.sum()
                migration_totals[scen_idx, sim_idx, y_idx, 2] = immigration.sum() - applied_emigration.sum()

                current_df = next_df
                save_classic_state(engine, current_df, scen_idx, sim_idx, y_idx + 1, total_pop, oadr, pop_age_sex)

    return {
        "population_total_summary": pd.concat(
            [engine.summarize_simulations(total_pop[i], state_years, scenarios[i], "total_population") for i in range(len(scenarios))],
            ignore_index=True,
        ),
        "births_summary": pd.concat(
            [engine.summarize_simulations(births[i], transition_years, scenarios[i], "births") for i in range(len(scenarios))],
            ignore_index=True,
        ),
        "old_age_dependency_summary": pd.concat(
            [engine.summarize_simulations(oadr[i], state_years, scenarios[i], "old_age_dependency_ratio") for i in range(len(scenarios))],
            ignore_index=True,
        ),
        "migration_summary": engine.summarize_migration_totals(migration_totals, transition_years, scenarios),
        "total_tfr_summary": pd.concat(
            [engine.summarize_simulations(total_tfr[i], transition_years, scenarios[i], "total_tfr") for i in range(len(scenarios))],
            ignore_index=True,
        ),
        "population_by_age_mean": engine.make_population_by_age_mean(pop_age_sex, state_years, scenarios),
        "tfr_summary": fertility["tfr_summary"],
        "e0_summary": mortality["e0_summary"],
        "initial_population_year": initial_population_year,
        "initial_population_df": initial_population_df,
        "mortality_version": mortality_version,
        "projection_style": "classic_dataframe_age_survive_migrate_birth_age_forward",
    }


def deterministic_population_path_classic(engine, components: dict, scenario) -> pd.DataFrame:
    """Central deterministic path used by the reverse calculator."""

    current_df = state_array_to_classic_df(engine, components["initial_state"])
    rows = [{"year": int(components["state_years"][0]), "population": total_population_classic(current_df)}]
    for y_idx, year in enumerate(components["transition_years"]):
        px = components["mortality"]["px_central"][y_idx, :, :]
        asfr = components["fertility"]["asfr_central"][y_idx, :, :]
        base_immigration = components["migration"]["single_age_mean"][engine.FLOW_TO_INDEX["immigration"], y_idx, :, :, :]
        base_emigration = components["migration"]["single_age_mean"][engine.FLOW_TO_INDEX["emigration"], y_idx, :, :, :]
        immigration, emigration = engine.apply_migration_scenario(
            base_immigration,
            base_emigration,
            scenario,
            int(year),
            components["initial_state"],
        )
        survived_df = apply_survival_classic(engine, current_df, px)
        migrated_df, _ = apply_migration_classic(engine, survived_df, immigration, emigration)
        _, births_by_citizenship = compute_births_classic(engine, migrated_df, asfr)
        current_df = age_forward_classic(engine, migrated_df, births_by_citizenship)
        rows.append({"year": int(year + 1), "population": total_population_classic(current_df)})
    return pd.DataFrame(rows)


def build_central_components_classic(engine, data_dir: Path, start_year: int, end_year: int, mortality_version: str, tfr_increase_pct: float) -> dict:
    """Build one central component set for the reverse search."""

    rng = np.random.default_rng(engine.RANDOM_SEED)
    transition_years = np.arange(start_year, end_year, dtype=int)
    state_years = np.arange(start_year, end_year + 1, dtype=int)
    initial_state, _, _ = engine.load_initial_population(data_dir, start_year)
    return {
        "transition_years": transition_years,
        "state_years": state_years,
        "initial_state": initial_state,
        "fertility": engine.build_fertility_forecast(data_dir, transition_years, 1, rng, tfr_increase_pct),
        "mortality": engine.build_mortality_forecast(data_dir, transition_years, 1, rng, mortality_version),
        "migration": engine.build_migration_forecast(data_dir, transition_years, 1, rng, initial_state),
    }


def population_at_target_classic(engine, components: dict, scenario, target_year: int) -> float:
    """Central projected population in target year."""

    path = deterministic_population_path_classic(engine, components, scenario)
    return float(path.loc[path["year"].eq(int(target_year)), "population"].iloc[0])


def reverse_search_classic(
    engine,
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
    """Required annual extra immigrants under the classic dataframe engine."""

    components = build_central_components_classic(engine, data_dir, start_year, end_year, mortality_version, tfr_increase_pct)
    policy_scenarios = [
        engine.Scenario(
            name="higher_total_immigration",
            label="Policy: higher total immigration only",
            policy_start_year=policy_start_year,
            policy_end_year=policy_end_year,
            adult_min_age=adult_min_age,
            adult_max_age=adult_max_age,
        ),
        engine.Scenario(
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
        base_value = population_at_target_classic(engine, components, scenario, target_year)
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
            trial = engine.Scenario(**{**scenario.__dict__, "extra_immigration_per_year": hi})
            if population_at_target_classic(engine, components, trial, target_year) >= target_population:
                break
            lo = hi
            hi *= 2.0

        if hi >= upper_limit:
            trial = engine.Scenario(**{**scenario.__dict__, "extra_immigration_per_year": upper_limit})
            achieved = population_at_target_classic(engine, components, trial, target_year)
            rows.append(
                {
                    "scenario": scenario.label,
                    "target_year": int(target_year),
                    "target_population": float(target_population),
                    "required_additional_immigrants_per_year": np.nan,
                    "achieved_population": achieved,
                    "gap_after_solution": achieved - target_population,
                    "policy_years": int(policy_end_year - policy_start_year + 1),
                    "status": f"Not reached below {upper_limit:,.0f} extra immigrants/year",
                }
            )
            continue

        for _ in range(28):
            mid = (lo + hi) / 2.0
            trial = engine.Scenario(**{**scenario.__dict__, "extra_immigration_per_year": mid})
            if population_at_target_classic(engine, components, trial, target_year) >= target_population:
                hi = mid
            else:
                lo = mid

        solved = engine.Scenario(**{**scenario.__dict__, "extra_immigration_per_year": hi})
        achieved = population_at_target_classic(engine, components, solved, target_year)
        rows.append(
            {
                "scenario": scenario.label,
                "target_year": int(target_year),
                "target_population": float(target_population),
                "required_additional_immigrants_per_year": float(hi),
                "achieved_population": float(achieved),
                "gap_after_solution": float(achieved - target_population),
                "policy_years": int(policy_end_year - policy_start_year + 1),
                "status": "Solved",
            }
        )

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=True)
def run_classic_cached(
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
    """Cached Streamlit wrapper around the classic dataframe projection."""

    engine = load_source_app()
    scenarios_all = engine.make_scenarios(
        policy_start_year=policy_start_year,
        policy_end_year=policy_end_year,
        extra_immigration_per_year=extra_immigration_per_year,
        child_share_of_extra=child_share_of_extra,
        adult_min_age=adult_min_age,
        adult_max_age=adult_max_age,
    )
    scenarios = [s for s in scenarios_all if s.name in set(selected_names)]
    return run_projection_classic(
        engine=engine,
        data_dir=Path(data_dir_str),
        start_year=int(start_year),
        end_year=int(end_year),
        n_sim=int(n_sim),
        mortality_version=mortality_version,
        scenarios=scenarios,
        tfr_increase_pct=float(tfr_increase_pct),
    )


@st.cache_data(show_spinner=True)
def reverse_classic_cached(
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
    """Cached reverse calculator for the classic engine."""

    engine = load_source_app()
    return reverse_search_classic(
        engine=engine,
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


def main() -> None:
    """Streamlit interface, in the order of the early prototype."""

    st.set_page_config(page_title="Italy projection - classic cohort component", layout="wide")
    st.title("Population Projection Simulator")
    st.caption(
        "Mixed version: raw-input component forecasts from the newer app, but the projection step follows the classic "
        "age-survive-migrate-birth-age-forward dataframe logic of the early prototype."
    )

    engine = load_source_app()
    scenario_options = {
        "baseline_migration": "Baseline migration scenario",
        "no_immigration": "No immigration scenario",
        "higher_total_immigration": "Higher total immigration only",
        "higher_immigration_younger_with_children": "Higher immigration + younger age structure + children",
    }

    st.sidebar.header("Input folder")
    data_dir = Path(st.sidebar.text_input("Repository/input folder", value=str(APP_DIR))).expanduser()

    st.sidebar.header("Projection Settings")
    start_year = st.sidebar.number_input("Start year", min_value=2000, max_value=2100, value=2025, step=1)
    end_year = st.sidebar.number_input("End year", min_value=int(start_year) + 1, max_value=2150, value=2075, step=1)
    n_sim = st.sidebar.slider("Simulation paths", min_value=10, max_value=200, value=40, step=10)
    mortality_version = st.sidebar.selectbox("Mortality version", ["excluding_covid", "all_years"], index=0)
    tfr_increase_pct = st.sidebar.slider("Increase forecast TFR by", 0.0, 50.0, 0.0, 1.0, format="%.0f%%")

    st.sidebar.header("Scenario Selection")
    selected_names = st.sidebar.multiselect(
        "Scenarios to display",
        options=list(scenario_options.keys()),
        default=list(scenario_options.keys()),
        format_func=lambda x: scenario_options[x],
    )

    st.sidebar.header("Immigration Policy Settings")
    extra_immigration_per_year = st.sidebar.number_input(
        "Additional immigrants per year",
        min_value=0,
        max_value=500_000,
        value=0,
        step=25_000,
    )
    policy_start_year = st.sidebar.number_input(
        "Policy start year",
        min_value=int(start_year),
        max_value=int(end_year) - 1,
        value=max(int(start_year), 2026),
        step=1,
    )
    policy_end_year = st.sidebar.number_input(
        "Policy end year",
        min_value=int(policy_start_year),
        max_value=int(end_year) - 1,
        value=int(end_year) - 1,
        step=1,
    )
    adult_min_age = st.sidebar.number_input("Minimum age of policy immigrants", value=20, min_value=0, max_value=100)
    adult_max_age = st.sidebar.number_input("Maximum age of policy immigrants", value=max(34, int(adult_min_age)), min_value=int(adult_min_age), max_value=100)
    child_share_of_extra = st.sidebar.slider("Child share of extra immigrants", 0.0, 0.60, 0.25, 0.05)

    if not selected_names:
        st.warning("Select at least one scenario.")
        st.stop()

    try:
        results = run_classic_cached(
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

    total = results["population_total_summary"]
    births = results["births_summary"]
    oadr = results["old_age_dependency_summary"]
    migration = results["migration_summary"]
    total_tfr = results["total_tfr_summary"]
    pop_mean = results["population_by_age_mean"]
    migration_benchmarks = engine.load_migration_benchmarks(data_dir)
    official_population_benchmarks = engine.load_official_population_benchmarks(data_dir)
    tfr_benchmarks = engine.load_tfr_benchmarks(data_dir)
    e0_benchmarks = engine.load_e0_benchmarks(data_dir)

    final_year = int(total["year"].max())
    baseline_final = total[(total["scenario"].eq("baseline_migration")) & (total["year"].eq(final_year))]
    c1, c2, c3, c4 = st.columns(4)
    if len(baseline_final):
        c1.metric(f"Baseline population {final_year}", f"{baseline_final['p50'].iloc[0]:,.0f}")
    c2.metric("Simulation paths", f"{n_sim}")
    c3.metric("Initial population year", str(results["initial_population_year"]))
    c4.metric("Projection style", "classic")

    st.subheader("Total Population Over Time")
    st.pyplot(
        engine.plot_total_population(
            total,
            list(selected_names),
            int(policy_start_year),
            int(policy_end_year),
            official_population_benchmarks,
        ),
        clear_figure=True,
    )

    st.subheader("Total Births Over Time")
    st.pyplot(
        engine.plot_summary_lines(
            births,
            list(selected_names),
            "Projected births",
            "Births",
            int(policy_start_year),
            int(policy_end_year),
        ),
        clear_figure=True,
    )

    st.subheader("Population Pyramid")
    pyramid_year = st.slider("Pyramid year", int(start_year), int(end_year), int(end_year))
    pyramid_scenario = st.selectbox(
        "Pyramid scenario",
        options=list(selected_names),
        format_func=lambda x: scenario_options[x],
    )
    st.pyplot(engine.plot_pyramid(pop_mean, pyramid_scenario, int(pyramid_year)), clear_figure=True)

    st.subheader("Old-Age Dependency Ratio (OADR) Over Time")
    st.pyplot(
        engine.plot_summary_lines(
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
        engine.plot_migration_with_benchmark(
            migration,
            migration_benchmarks,
            list(selected_names),
            int(policy_start_year),
            int(policy_end_year),
        ),
        clear_figure=True,
    )

    st.subheader("Total TFR")
    st.pyplot(
        engine.plot_total_tfr(
            total_tfr,
            list(selected_names),
            tfr_benchmarks,
            int(policy_start_year),
            int(policy_end_year),
        ),
        clear_figure=True,
    )

    st.subheader("Life expectancy at birth")
    st.pyplot(
        engine.plot_e0(
            results["e0_summary"],
            e0_benchmarks,
            int(policy_start_year),
            int(policy_end_year),
        ),
        clear_figure=True,
    )

    st.subheader("Reverse policy calculator")
    st.caption(
        "Uses the same classic dataframe projection engine with central component paths."
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
        reverse = reverse_classic_cached(
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

    with st.expander("Download result tables"):
        st.download_button(
            "Download total population summary",
            total.to_csv(index=False).encode("utf-8"),
            file_name="classic_population_total_summary.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download population by age mean",
            pop_mean.to_csv(index=False).encode("utf-8"),
            file_name="classic_population_by_age_mean.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download migration summary",
            migration.to_csv(index=False).encode("utf-8"),
            file_name="classic_migration_summary.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download total TFR summary",
            total_tfr.to_csv(index=False).encode("utf-8"),
            file_name="classic_total_tfr_summary.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
