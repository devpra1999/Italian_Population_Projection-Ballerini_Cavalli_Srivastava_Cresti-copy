"""
mix_normal.py

Normal Python version of the mixed cohort-component projection.

This script is meant to be easy to read and easy to share with someone who is
new to Python.  It follows the style of `Projections_commented copy.py`: the
projection is written step by step, with comments explaining what each block is
doing.

Important:
    - This is NOT a Streamlit app.
    - It creates output files directly under output/mix_normal/.
    - It uses the same raw inputs, component assumptions, scenarios, benchmark
      loaders, and plotting helpers used by `mix_streamlit.py`.
    - The projection itself is written explicitly:
        take population at age x,
        multiply by survival probability,
        subtract emigration,
        add immigration,
        calculate births,
        age everyone forward,
        put newborns at age 0.

How to run:
    python3 mix_normal.py
"""

from __future__ import annotations

# =============================================================================
# 0. Import packages
# =============================================================================

import importlib.util  # Used to import the helper app even though its filename has spaces.
import os             # Used to put local caches inside the project folder.
import sys            # Used to register imported modules safely.
from pathlib import Path

import numpy as np
import pandas as pd

# Matplotlib sometimes tries to write a font cache under the user's home folder.
# That can be annoying when the script is moved to another computer, GitHub
# runtime, or restricted environment.  We set a local cache folder before
# importing matplotlib, so figures can be produced without home-folder warnings.
LOCAL_CACHE_DIR = Path(__file__).resolve().parent / ".cache"
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE_DIR))

import matplotlib.pyplot as plt


# =============================================================================
# 1. User settings
# =============================================================================

# Project folder.  Because this script is saved in projection_project, parent is
# the folder where all raw input files are expected to live.
PROJECT_DIR = Path(__file__).resolve().parent

# Raw input folder.  Usually this is the same as PROJECT_DIR.
DATA_DIR = PROJECT_DIR

# Output folder.  The script creates this folder if it does not exist.
OUTPUT_DIR = PROJECT_DIR / "output" / "mix_normal"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Projection horizon.
START_YEAR = 2025
END_YEAR = 2075

# Number of stochastic simulation paths.
# Increase this for smoother uncertainty intervals; reduce it for faster tests.
N_SIM = 40

# Mortality version.
# "excluding_covid" ignores COVID years when fitting the mortality trend.
# "all_years" keeps all observed years.
MORTALITY_VERSION = "excluding_covid"

# Optional increase in forecast TFR.  Example: 10 means all forecast ASFR/TFR
# levels are multiplied by 1.10.
TFR_INCREASE_PCT = 0.0

# Policy years and policy migration settings.
POLICY_START_YEAR = 2026
POLICY_END_YEAR = 2074
EXTRA_IMMIGRANTS_PER_YEAR = 0
CHILD_SHARE_OF_EXTRA = 0.25
POLICY_ADULT_MIN_AGE = 20
POLICY_ADULT_MAX_AGE = 34

# Scenarios to include in the output.
SELECTED_SCENARIOS = [
    "baseline_migration",
    "no_immigration",
    "higher_total_immigration",
    "higher_immigration_younger_with_children",
]

# Reverse calculator settings.
# The script will estimate how many extra immigrants per year are needed to
# reach this target population in this target year.
TARGET_YEAR = END_YEAR
TARGET_POPULATION = 50_000_000


# =============================================================================
# 2. Load helper modules
# =============================================================================

# The file `streamlit_projection_app copy.py` already contains robust readers for
# raw input files, forecast builders for fertility/mortality/migration, benchmark
# loaders, and plotting functions.  We reuse those pieces so the method remains
# aligned with the app.
HELPER_APP_PATH = PROJECT_DIR / "streamlit_projection_app copy.py"

# The file `mix_streamlit.py` contains the same classic projection helper logic
# used in the Streamlit mixed version.  We only use it here for the reverse
# immigration calculator at the end.
MIX_STREAMLIT_PATH = PROJECT_DIR / "mix_streamlit.py"

# Import `streamlit_projection_app copy.py` as a normal Python module.
spec = importlib.util.spec_from_file_location("projection_helper_app", HELPER_APP_PATH)
engine = importlib.util.module_from_spec(spec)
sys.modules["projection_helper_app"] = engine
spec.loader.exec_module(engine)

# Import `mix_streamlit.py` as a normal Python module.
spec_mix = importlib.util.spec_from_file_location("mix_streamlit_helpers", MIX_STREAMLIT_PATH)
mix_helpers = importlib.util.module_from_spec(spec_mix)
sys.modules["mix_streamlit_helpers"] = mix_helpers
spec_mix.loader.exec_module(mix_helpers)


# =============================================================================
# 3. Build scenario objects
# =============================================================================

# Scenario objects store the settings used later inside the migration scenario
# modifier.  We create all available scenarios, then keep only the selected ones.
all_scenarios = engine.make_scenarios(
    policy_start_year=POLICY_START_YEAR,
    policy_end_year=POLICY_END_YEAR,
    extra_immigration_per_year=EXTRA_IMMIGRANTS_PER_YEAR,
    child_share_of_extra=CHILD_SHARE_OF_EXTRA,
    adult_min_age=POLICY_ADULT_MIN_AGE,
    adult_max_age=POLICY_ADULT_MAX_AGE,
)

scenarios = [scenario for scenario in all_scenarios if scenario.name in SELECTED_SCENARIOS]


# =============================================================================
# 4. Build component forecasts from raw input files
# =============================================================================

# A fixed random seed makes the stochastic paths reproducible.
rng = np.random.default_rng(engine.RANDOM_SEED)

# transition_years are years where we move from year t to year t+1.
# Example: if START_YEAR=2025 and END_YEAR=2075, transition_years are 2025-2074.
transition_years = np.arange(START_YEAR, END_YEAR, dtype=int)

# state_years are years where we store a population stock.
# Example: 2025-2075 inclusive.
state_years = np.arange(START_YEAR, END_YEAR + 1, dtype=int)

# Load the observed initial population.  The result is a 3D array:
#     citizenship x sex x age
initial_state, initial_population_df, initial_population_year = engine.load_initial_population(DATA_DIR, START_YEAR)

# Build fertility forecasts:
#     ASFR by simulation x year x citizenship x age,
#     TFR summary tables.
fertility = engine.build_fertility_forecast(
    data_dir=DATA_DIR,
    years=transition_years,
    n_sim=N_SIM,
    rng=rng,
    tfr_increase_pct=TFR_INCREASE_PCT,
)

# Build mortality/survival forecasts:
#     survival probabilities by simulation x year x sex x age,
#     e0 summary tables.
mortality = engine.build_mortality_forecast(
    data_dir=DATA_DIR,
    years=transition_years,
    n_sim=N_SIM,
    rng=rng,
    mortality_version=MORTALITY_VERSION,
)

# Build migration forecasts:
#     immigration and emigration by simulation x flow x year x citizenship x sex x age.
migration = engine.build_migration_forecast(
    data_dir=DATA_DIR,
    years=transition_years,
    n_sim=N_SIM,
    rng=rng,
    initial_state=initial_state,
)


# =============================================================================
# 5. Prepare arrays where we store results
# =============================================================================

# Number of scenarios, simulations, state years and transition years.
n_scenarios = len(scenarios)
n_state_years = len(state_years)
n_transition_years = len(transition_years)

# total_pop will store total population by scenario x simulation x year.
total_pop = np.zeros((n_scenarios, N_SIM, n_state_years), dtype=float)

# births will store total births by scenario x simulation x transition year.
births = np.zeros((n_scenarios, N_SIM, n_transition_years), dtype=float)

# oadr will store old-age dependency ratio by scenario x simulation x year.
oadr = np.zeros((n_scenarios, N_SIM, n_state_years), dtype=float)

# total_tfr will store one total-population TFR by scenario x simulation x year.
total_tfr = np.zeros((n_scenarios, N_SIM, n_transition_years), dtype=float)

# pop_age_sex will store population by sex and age.  Citizenship is summed
# because it is mainly used for pyramid plots.
pop_age_sex = np.zeros(
    (n_scenarios, N_SIM, n_state_years, len(engine.SEXES), len(engine.AGES)),
    dtype=float,
)

# migration_totals stores total immigration, total emigration and net migration.
migration_totals = np.zeros((n_scenarios, N_SIM, n_transition_years, 3), dtype=float)


# =============================================================================
# 6. Run the projection, explicitly and step by step
# =============================================================================

# The initial population array is easier for the component builders, but the
# classic projection is easier to see as a dataframe with one row per age.
#
# We will create this dataframe at the start of each simulation:
#     eta, maschi_italiani, femmine_italiani, maschi_stranieri, femmine_stranieri

for scen_idx, scenario in enumerate(scenarios):
    # Loop over migration policy scenarios.

    for sim_idx in range(N_SIM):
        # Loop over stochastic simulation paths.

        # ---------------------------------------------------------------------
        # 6.1 Start from the observed initial population
        # ---------------------------------------------------------------------

        current_df = pd.DataFrame(
            {
                "eta": engine.AGES,
                "maschi_italiani": initial_state[engine.CIT_TO_INDEX["italiani"], engine.SEX_TO_INDEX["male"], :],
                "femmine_italiani": initial_state[engine.CIT_TO_INDEX["italiani"], engine.SEX_TO_INDEX["female"], :],
                "maschi_stranieri": initial_state[engine.CIT_TO_INDEX["stranieri"], engine.SEX_TO_INDEX["male"], :],
                "femmine_stranieri": initial_state[engine.CIT_TO_INDEX["stranieri"], engine.SEX_TO_INDEX["female"], :],
            }
        )

        # Store population at the first year.
        total_pop[scen_idx, sim_idx, 0] = (
            current_df["maschi_italiani"].sum()
            + current_df["femmine_italiani"].sum()
            + current_df["maschi_stranieri"].sum()
            + current_df["femmine_stranieri"].sum()
        )

        working_pop = current_df.loc[current_df["eta"].between(15, 64), [
            "maschi_italiani",
            "femmine_italiani",
            "maschi_stranieri",
            "femmine_stranieri",
        ]].sum().sum()

        old_pop = current_df.loc[current_df["eta"].ge(65), [
            "maschi_italiani",
            "femmine_italiani",
            "maschi_stranieri",
            "femmine_stranieri",
        ]].sum().sum()

        oadr[scen_idx, sim_idx, 0] = 100.0 * old_pop / working_pop if working_pop > 0 else np.nan

        pop_age_sex[scen_idx, sim_idx, 0, engine.SEX_TO_INDEX["male"], :] = (
            current_df["maschi_italiani"].to_numpy(float)
            + current_df["maschi_stranieri"].to_numpy(float)
        )
        pop_age_sex[scen_idx, sim_idx, 0, engine.SEX_TO_INDEX["female"], :] = (
            current_df["femmine_italiani"].to_numpy(float)
            + current_df["femmine_stranieri"].to_numpy(float)
        )

        # ---------------------------------------------------------------------
        # 6.2 Move year by year
        # ---------------------------------------------------------------------

        for y_idx, year in enumerate(transition_years):
            # next_year is the population stock year produced by this iteration.
            next_year = year + 1

            # Pick this simulation's survival probabilities for this year.
            px = mortality["px_sim"][sim_idx, y_idx, :, :]

            # Pick this simulation's fertility rates for this year.
            asfr = fertility["asfr_sim"][sim_idx, y_idx, :, :]

            # Pick baseline migration by single age, sex and citizenship.
            base_immigration = migration["single_age_sim"][
                sim_idx,
                engine.FLOW_TO_INDEX["immigration"],
                y_idx,
                :, :, :,
            ]
            base_emigration = migration["single_age_sim"][
                sim_idx,
                engine.FLOW_TO_INDEX["emigration"],
                y_idx,
                :, :, :,
            ]

            # Apply scenario rules:
            # - baseline keeps baseline migration;
            # - no immigration sets immigration to zero;
            # - policy scenarios add extra immigrants.
            immigration, emigration = engine.apply_migration_scenario(
                base_immigration,
                base_emigration,
                scenario,
                int(year),
                initial_state,
            )

            # -----------------------------------------------------------------
            # Step A: calculate total TFR for this year
            # -----------------------------------------------------------------

            # Total TFR needs the current female age distribution by citizenship.
            # Here we rebuild a small array from current_df because the helper
            # expects citizenship x sex x age.
            current_state_for_tfr = np.zeros(
                (len(engine.CITIZENSHIPS), len(engine.SEXES), len(engine.AGES)),
                dtype=float,
            )
            current_state_for_tfr[engine.CIT_TO_INDEX["italiani"], engine.SEX_TO_INDEX["male"], :] = current_df["maschi_italiani"].to_numpy(float)
            current_state_for_tfr[engine.CIT_TO_INDEX["italiani"], engine.SEX_TO_INDEX["female"], :] = current_df["femmine_italiani"].to_numpy(float)
            current_state_for_tfr[engine.CIT_TO_INDEX["stranieri"], engine.SEX_TO_INDEX["male"], :] = current_df["maschi_stranieri"].to_numpy(float)
            current_state_for_tfr[engine.CIT_TO_INDEX["stranieri"], engine.SEX_TO_INDEX["female"], :] = current_df["femmine_stranieri"].to_numpy(float)

            total_tfr[scen_idx, sim_idx, y_idx] = engine.compute_total_tfr_from_state(current_state_for_tfr, asfr)

            # -----------------------------------------------------------------
            # Step B: apply survival probabilities
            # -----------------------------------------------------------------

            survived_df = current_df.copy()

            # Male survival probabilities are used for male columns.
            survived_df["maschi_italiani"] *= px[engine.SEX_TO_INDEX["male"], :]
            survived_df["maschi_stranieri"] *= px[engine.SEX_TO_INDEX["male"], :]

            # Female survival probabilities are used for female columns.
            survived_df["femmine_italiani"] *= px[engine.SEX_TO_INDEX["female"], :]
            survived_df["femmine_stranieri"] *= px[engine.SEX_TO_INDEX["female"], :]

            # -----------------------------------------------------------------
            # Step C: subtract emigration and add immigration
            # -----------------------------------------------------------------

            migrated_df = survived_df.copy()
            applied_emigration = np.zeros_like(emigration)

            # This dictionary tells Python which array cell corresponds to which
            # dataframe column.
            population_columns = {
                ("italiani", "male"): "maschi_italiani",
                ("italiani", "female"): "femmine_italiani",
                ("stranieri", "male"): "maschi_stranieri",
                ("stranieri", "female"): "femmine_stranieri",
            }

            for (citizenship, sex), column in population_columns.items():
                c = engine.CIT_TO_INDEX[citizenship]
                s = engine.SEX_TO_INDEX[sex]

                # Population after survival, before migration.
                current_population = migrated_df[column].to_numpy(float)

                # We cannot subtract more emigrants than the number of people in
                # that age-sex-citizenship cell.
                emigrants = np.minimum(emigration[c, s, :], current_population)

                # Store actual emigrants after the cap.
                applied_emigration[c, s, :] = emigrants

                # Migration accounting:
                #     population = survivors - emigrants + immigrants
                migrated_df[column] = current_population - emigrants + immigration[c, s, :]

            # -----------------------------------------------------------------
            # Step D: calculate births
            # -----------------------------------------------------------------

            # Births from Italian women.
            births_italiani = float(
                (
                    migrated_df["femmine_italiani"].to_numpy(float)
                    * asfr[engine.CIT_TO_INDEX["italiani"], :]
                ).sum()
            )

            # Births from foreign women.
            births_stranieri = float(
                (
                    migrated_df["femmine_stranieri"].to_numpy(float)
                    * asfr[engine.CIT_TO_INDEX["stranieri"], :]
                ).sum()
            )

            # Total births in this year.
            total_births = births_italiani + births_stranieri

            # Store births for output tables.
            births[scen_idx, sim_idx, y_idx] = total_births

            # -----------------------------------------------------------------
            # Step E: age everyone forward
            # -----------------------------------------------------------------

            next_df = pd.DataFrame({"eta": engine.AGES})

            for column in [
                "maschi_italiani",
                "femmine_italiani",
                "maschi_stranieri",
                "femmine_stranieri",
            ]:
                # values[0] is age 0, values[1] is age 1, ..., values[100] is 100.
                values = migrated_df[column].to_numpy(float)

                # Create an empty vector for next year.
                aged_values = np.zeros_like(values)

                # Everyone aged x becomes x+1 next year.
                aged_values[1:] = values[:-1]

                # Age 100 is open-ended: people already aged 100 remain in age 100.
                aged_values[100] += values[100]

                # Save the aged population.
                next_df[column] = aged_values

            # -----------------------------------------------------------------
            # Step F: insert newborns at age 0
            # -----------------------------------------------------------------

            # Split births by sex using the sex ratio at birth.
            next_df.loc[0, "maschi_italiani"] = births_italiani * engine.SEX_RATIO_AT_BIRTH_MALE
            next_df.loc[0, "femmine_italiani"] = births_italiani * (1.0 - engine.SEX_RATIO_AT_BIRTH_MALE)
            next_df.loc[0, "maschi_stranieri"] = births_stranieri * engine.SEX_RATIO_AT_BIRTH_MALE
            next_df.loc[0, "femmine_stranieri"] = births_stranieri * (1.0 - engine.SEX_RATIO_AT_BIRTH_MALE)

            # -----------------------------------------------------------------
            # Step G: save migration totals and next-year population indicators
            # -----------------------------------------------------------------

            migration_totals[scen_idx, sim_idx, y_idx, 0] = immigration.sum()
            migration_totals[scen_idx, sim_idx, y_idx, 1] = applied_emigration.sum()
            migration_totals[scen_idx, sim_idx, y_idx, 2] = immigration.sum() - applied_emigration.sum()

            total_pop[scen_idx, sim_idx, y_idx + 1] = (
                next_df["maschi_italiani"].sum()
                + next_df["femmine_italiani"].sum()
                + next_df["maschi_stranieri"].sum()
                + next_df["femmine_stranieri"].sum()
            )

            working_pop = next_df.loc[next_df["eta"].between(15, 64), [
                "maschi_italiani",
                "femmine_italiani",
                "maschi_stranieri",
                "femmine_stranieri",
            ]].sum().sum()

            old_pop = next_df.loc[next_df["eta"].ge(65), [
                "maschi_italiani",
                "femmine_italiani",
                "maschi_stranieri",
                "femmine_stranieri",
            ]].sum().sum()

            oadr[scen_idx, sim_idx, y_idx + 1] = 100.0 * old_pop / working_pop if working_pop > 0 else np.nan

            pop_age_sex[scen_idx, sim_idx, y_idx + 1, engine.SEX_TO_INDEX["male"], :] = (
                next_df["maschi_italiani"].to_numpy(float)
                + next_df["maschi_stranieri"].to_numpy(float)
            )
            pop_age_sex[scen_idx, sim_idx, y_idx + 1, engine.SEX_TO_INDEX["female"], :] = (
                next_df["femmine_italiani"].to_numpy(float)
                + next_df["femmine_stranieri"].to_numpy(float)
            )

            # The next year becomes the current year for the next loop.
            current_df = next_df.copy()


# =============================================================================
# 7. Convert simulation arrays into readable output tables
# =============================================================================

population_total_summary = pd.concat(
    [
        engine.summarize_simulations(total_pop[i], state_years, scenarios[i], "total_population")
        for i in range(n_scenarios)
    ],
    ignore_index=True,
)

births_summary = pd.concat(
    [
        engine.summarize_simulations(births[i], transition_years, scenarios[i], "births")
        for i in range(n_scenarios)
    ],
    ignore_index=True,
)

old_age_dependency_summary = pd.concat(
    [
        engine.summarize_simulations(oadr[i], state_years, scenarios[i], "old_age_dependency_ratio")
        for i in range(n_scenarios)
    ],
    ignore_index=True,
)

migration_summary = engine.summarize_migration_totals(migration_totals, transition_years, scenarios)

total_tfr_summary = pd.concat(
    [
        engine.summarize_simulations(total_tfr[i], transition_years, scenarios[i], "total_tfr")
        for i in range(n_scenarios)
    ],
    ignore_index=True,
)

population_by_age_mean = engine.make_population_by_age_mean(pop_age_sex, state_years, scenarios)


# =============================================================================
# 8. Save tables
# =============================================================================

population_total_summary.to_csv(TABLE_DIR / "population_total_summary.csv", index=False)
births_summary.to_csv(TABLE_DIR / "births_summary.csv", index=False)
old_age_dependency_summary.to_csv(TABLE_DIR / "old_age_dependency_summary.csv", index=False)
migration_summary.to_csv(TABLE_DIR / "migration_summary.csv", index=False)
total_tfr_summary.to_csv(TABLE_DIR / "total_tfr_summary.csv", index=False)
population_by_age_mean.to_csv(TABLE_DIR / "population_by_age_mean.csv", index=False)
fertility["tfr_summary"].to_csv(TABLE_DIR / "tfr_by_citizenship_summary.csv", index=False)
mortality["e0_summary"].to_csv(TABLE_DIR / "e0_summary.csv", index=False)


# =============================================================================
# 9. Load benchmark data for plots
# =============================================================================

migration_benchmarks = engine.load_migration_benchmarks(DATA_DIR)
official_population_benchmarks = engine.load_official_population_benchmarks(DATA_DIR)
tfr_benchmarks = engine.load_tfr_benchmarks(DATA_DIR)
e0_benchmarks = engine.load_e0_benchmarks(DATA_DIR)


# =============================================================================
# 10. Make and save figures
# =============================================================================

# Total population.
fig = engine.plot_total_population(
    population_total_summary,
    SELECTED_SCENARIOS,
    POLICY_START_YEAR,
    POLICY_END_YEAR,
    official_population_benchmarks,
)
fig.savefig(FIGURE_DIR / "projected_total_population.png", dpi=220)
plt.close(fig)

# Births.
fig = engine.plot_summary_lines(
    births_summary,
    SELECTED_SCENARIOS,
    "Projected births",
    "Births",
    POLICY_START_YEAR,
    POLICY_END_YEAR,
)
fig.savefig(FIGURE_DIR / "projected_births.png", dpi=220)
plt.close(fig)

# Old-age dependency ratio.
fig = engine.plot_summary_lines(
    old_age_dependency_summary,
    SELECTED_SCENARIOS,
    "Old-age dependency ratio",
    "Population 65+ / population 15-64 (%)",
    POLICY_START_YEAR,
    POLICY_END_YEAR,
)
fig.savefig(FIGURE_DIR / "old_age_dependency_ratio.png", dpi=220)
plt.close(fig)

# Net migration.
fig = engine.plot_migration_with_benchmark(
    migration_summary,
    migration_benchmarks,
    SELECTED_SCENARIOS,
    POLICY_START_YEAR,
    POLICY_END_YEAR,
)
fig.savefig(FIGURE_DIR / "net_migration.png", dpi=220)
plt.close(fig)

# Total TFR.
fig = engine.plot_total_tfr(
    total_tfr_summary,
    SELECTED_SCENARIOS,
    tfr_benchmarks,
    POLICY_START_YEAR,
    POLICY_END_YEAR,
)
fig.savefig(FIGURE_DIR / "total_tfr.png", dpi=220)
plt.close(fig)

# Life expectancy at birth.
fig = engine.plot_e0(
    mortality["e0_summary"],
    e0_benchmarks,
    POLICY_START_YEAR,
    POLICY_END_YEAR,
)
fig.savefig(FIGURE_DIR / "life_expectancy_e0.png", dpi=220)
plt.close(fig)

# Population pyramid for the final year, baseline scenario.
fig = engine.plot_pyramid(population_by_age_mean, "baseline_migration", END_YEAR)
fig.savefig(FIGURE_DIR / f"population_pyramid_baseline_{END_YEAR}.png", dpi=220)
plt.close(fig)


# =============================================================================
# 11. Reverse immigration calculator
# =============================================================================

# This part answers:
#     "If I want population X in year Y, how many extra immigrants per year
#      are needed during the policy period?"
#
# The reverse calculator uses central component paths to keep the result
# transparent and fast.
reverse_required_immigration = mix_helpers.reverse_search_classic(
    engine=engine,
    data_dir=DATA_DIR,
    start_year=START_YEAR,
    end_year=END_YEAR,
    mortality_version=MORTALITY_VERSION,
    tfr_increase_pct=TFR_INCREASE_PCT,
    target_population=TARGET_POPULATION,
    target_year=TARGET_YEAR,
    policy_start_year=POLICY_START_YEAR,
    policy_end_year=POLICY_END_YEAR,
    child_share_of_extra=CHILD_SHARE_OF_EXTRA,
    adult_min_age=POLICY_ADULT_MIN_AGE,
    adult_max_age=POLICY_ADULT_MAX_AGE,
)

reverse_required_immigration.to_csv(TABLE_DIR / "reverse_required_immigration.csv", index=False)


# =============================================================================
# 12. Print a short summary
# =============================================================================

selected_years = [START_YEAR, 2030, 2050, END_YEAR]
selected_years = [year for year in selected_years if START_YEAR <= year <= END_YEAR]

print("\nProjection finished.")
print(f"Initial population year used: {initial_population_year}")
print(f"Tables saved in:  {TABLE_DIR}")
print(f"Figures saved in: {FIGURE_DIR}")
print("\nTotal population summary for selected years:")
print(
    population_total_summary.loc[
        population_total_summary["year"].isin(selected_years),
        ["scenario", "year", "p10", "p50", "p90"],
    ].to_string(index=False)
)

print("\nReverse immigration calculator:")
print(reverse_required_immigration.to_string(index=False))
