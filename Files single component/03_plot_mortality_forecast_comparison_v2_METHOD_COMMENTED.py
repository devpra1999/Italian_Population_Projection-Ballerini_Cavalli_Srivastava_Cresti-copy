# =============================================================================
# MORTALITY MODULE - FIGURES AND BENCHMARK CHECKS
# =============================================================================
# Reading guide
# -------------
# This script reads mortality outputs from 02 and produces comparison figures.
# It is intentionally separate from estimation: the model creates mortality
# schedules and life expectancy series, while this script checks whether those
# outputs look plausible against external benchmarks.
# =============================================================================

"""
03_plot_mortality_forecast_v2.py

Figures for the mortality module.

Reads the two model versions produced by:
    02_fit_forecast_mortality_leemiller_v2.py

Figures written to:
    output/mortality/figures/

Optional benchmark file:
    output/mortality/comparison/life_expectancy_benchmarks_e0.csv
or:
    data/processed/mortality_benchmarks_e0.csv

Benchmark expected columns:
    source, year, sex, e0
Optional:
    e0_lower, e0_upper, scenario
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_DIR = Path("/Users/andreaballerini/Downloads/projection_project")
OUT = PROJECT_DIR / "output" / "mortality"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

MODEL_VERSIONS = ["all_years", "excluding_covid"]


def read_if_exists(path: Path) -> pd.DataFrame:
    """
    Read a model output only if it exists. This lets the plotting script run even when some
    optional comparison files are missing.
    """
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_model_outputs(model_version: str) -> dict:
    """
    Load all mortality model outputs needed for the figures.
    """
    d = OUT / model_version
    return {
        "age_params": read_if_exists(d / "lee_miller_age_parameters.csv"),
        "kt_hist": read_if_exists(d / "lee_miller_time_index_history.csv"),
        "kt_mean": read_if_exists(d / "lee_miller_time_index_forecast_mean.csv"),
        "kt_sims": read_if_exists(d / "lee_miller_time_index_forecast_simulations.csv"),
        "fit_qx": read_if_exists(d / "mortality_fit_observed_vs_fitted.csv"),
        "fut_sched": read_if_exists(d / "mortality_forecast_schedule_mean.csv"),
        "full_sched": read_if_exists(d / "mortality_history_plus_forecast_mean.csv"),
        "e0_hist": read_if_exists(d / "life_expectancy_history.csv"),
        "e0_mean": read_if_exists(d / "life_expectancy_forecast_mean.csv"),
        "e0_sims": read_if_exists(d / "life_expectancy_forecast_simulations.csv"),
        "e0_int": read_if_exists(d / "life_expectancy_forecast_intervals.csv"),
    }


def normalize_benchmark_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Make benchmark sex labels compatible with model labels.

    Needed because the e0 comparison file uses Italian labels:
    Uomini/Donne, while the model outputs use male/female.
    """
    if df.empty or "sex" not in df.columns:
        return df
    out = df.copy()
    mapping = {
        "m": "male", "maschi": "male", "maschio": "male", "male": "male", "men": "male", "uomini": "male", "uomo": "male",
        "f": "female", "femmine": "female", "femmina": "female", "female": "female", "women": "female", "donne": "female", "donna": "female",
        "totale": "total", "total": "total", "both": "total",
    }
    out["sex"] = out["sex"].astype(str).str.strip().str.lower().map(mapping).fillna(out["sex"].astype(str).str.strip().str.lower())
    return out


def load_benchmarks() -> pd.DataFrame:
    """
    Load mortality benchmark data used later for validation, not for estimating our model.
    """
    candidates = [
        OUT / "comparison" / "life_expectancy_benchmarks_e0.csv",
        PROJECT_DIR / "data" / "processed" / "mortality_benchmarks_e0.csv",
        PROJECT_DIR / "mortality_benchmarks_e0.csv",
    ]
    for p in candidates:
        if p.exists():
            return normalize_benchmark_sex(pd.read_csv(p))
    return pd.DataFrame()


models = {mv: load_model_outputs(mv) for mv in MODEL_VERSIONS}
bench = load_benchmarks()

# 1. k_t observed and forecast for each model version
for mv, obj in models.items():
    kt_hist = obj["kt_hist"]
    kt_mean = obj["kt_mean"]
    if kt_hist.empty or kt_mean.empty:
        continue
    for sex in sorted(kt_hist["sex"].unique()):
        gh = kt_hist[kt_hist["sex"] == sex].copy()
        gf = kt_mean[kt_mean["sex"] == sex].copy()

        plt.figure(figsize=(9, 5))
        plt.plot(gh["year"], gh["kt"], linewidth=2, label="Observed / adjusted k_t")
        plt.plot(gf["year"], gf["kt"], linewidth=2, linestyle="--", label="Forecast k_t")
        plt.title(f"Lee-Miller mortality index - {sex} - {mv}")
        plt.xlabel("Year")
        plt.ylabel("k_t")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(FIG / f"fig1_kt_{mv}_{sex}.png", dpi=220)
        plt.savefig(FIG / f"fig1_kt_{mv}_{sex}.pdf")
        plt.close()

# 2. life expectancy fan chart by model version
for mv, obj in models.items():
    e0_int = obj["e0_int"]
    e0_hist = obj["e0_hist"]
    if e0_int.empty:
        continue
    for sex in sorted(e0_int["sex"].unique()):
        gf = e0_int[e0_int["sex"] == sex].copy()
        gh = e0_hist[e0_hist["sex"] == sex].copy() if not e0_hist.empty else pd.DataFrame()

        plt.figure(figsize=(9, 5))
        if not gh.empty:
            plt.plot(gh["year"], gh["e0_observed"], linewidth=2, label="Observed e0 used in fit")
        plt.fill_between(gf["year"], gf["p025"], gf["p975"], alpha=0.16, label="2.5-97.5% interval")
        plt.fill_between(gf["year"], gf["p10"], gf["p90"], alpha=0.22, label="10-90% interval")
        plt.plot(gf["year"], gf["mean"], linewidth=2, label="Mean forecast")
        plt.plot(gf["year"], gf["p50"], linewidth=1.5, linestyle="--", label="Median simulation")
        plt.title(f"Life expectancy at birth forecast - {sex} - {mv}")
        plt.xlabel("Year")
        plt.ylabel("e0")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(FIG / f"fig2_e0_fanchart_{mv}_{sex}.png", dpi=220)
        plt.savefig(FIG / f"fig2_e0_fanchart_{mv}_{sex}.pdf")
        plt.close()

# 3. direct comparison: all years vs excluding COVID, plus benchmark lines if available
comp = read_if_exists(OUT / "comparison" / "life_expectancy_forecast_intervals_all_models.csv")
hist_comp = read_if_exists(OUT / "comparison" / "life_expectancy_history_all_models.csv")
if not comp.empty:
    for sex in sorted(comp["sex"].unique()):
        plt.figure(figsize=(10, 6))
        gh = hist_comp[(hist_comp["sex"] == sex) & (hist_comp["model_version"] == "all_years")].copy() if not hist_comp.empty else pd.DataFrame()
        if not gh.empty:
            plt.plot(gh["year"], gh["e0_observed"], linewidth=2, label="Observed e0")

        for mv in MODEL_VERSIONS:
            gf = comp[(comp["sex"] == sex) & (comp["model_version"] == mv)].copy()
            if gf.empty:
                continue
            plt.plot(gf["year"], gf["mean"], linewidth=2, label=f"{mv}: mean")
            plt.plot(gf["year"], gf["p50"], linewidth=1.4, linestyle="--", label=f"{mv}: median")

        if not bench.empty:
            gb = bench[bench["sex"] == sex].copy()
            for source, gs in gb.groupby("source"):
                gs = gs.sort_values("year")
                plt.plot(gs["year"], gs["e0"], linewidth=1.8, linestyle=":", label=f"Benchmark: {source}")

        plt.title(f"Life expectancy at birth: model comparison and benchmarks - {sex}")
        plt.xlabel("Year")
        plt.ylabel("e0")
        plt.legend(frameon=False, ncol=2)
        plt.tight_layout()
        plt.savefig(FIG / f"fig3_e0_model_comparison_benchmarks_{sex}.png", dpi=220)
        plt.savefig(FIG / f"fig3_e0_model_comparison_benchmarks_{sex}.pdf")
        plt.close()

# 4. age parameters ax and bx
for mv, obj in models.items():
    age_params = obj["age_params"]
    if age_params.empty:
        continue
    for sex in sorted(age_params["sex"].unique()):
        g = age_params[age_params["sex"] == sex].copy()
        plt.figure(figsize=(10, 6))
        plt.plot(g["age"], g["ax"], linewidth=2, label="a_x")
        plt.plot(g["age"], g["bx"], linewidth=2, label="b_x")
        plt.title(f"Lee-Miller age parameters - {sex} - {mv}")
        plt.xlabel("Age")
        plt.ylabel("Parameter value")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(FIG / f"fig4_age_parameters_{mv}_{sex}.png", dpi=220)
        plt.savefig(FIG / f"fig4_age_parameters_{mv}_{sex}.pdf")
        plt.close()

# 5. mortality schedules observed vs projected
for mv, obj in models.items():
    fut_sched = obj["fut_sched"]
    full_sched = obj["full_sched"]
    if fut_sched.empty or full_sched.empty:
        continue

    years = sorted(fut_sched["year"].unique())
    pick_years = [years[0], years[min(9, len(years) - 1)], years[min(24, len(years) - 1)], years[-1]]
    pick_years = sorted(set(pick_years))

    for sex in sorted(full_sched["sex"].unique()):
        plt.figure(figsize=(10, 6))
        gh = full_sched[(full_sched["sex"] == sex) & (full_sched["source"] == "observed")]
        if len(gh):
            for y in sorted(gh["year"].unique())[-3:]:
                tmp = gh[gh["year"] == y].sort_values("age")
                plt.plot(tmp["age"], tmp["qx"], linewidth=1.8, label=f"Observed {y}")
        gf = fut_sched[fut_sched["sex"] == sex]
        for y in pick_years:
            tmp = gf[gf["year"] == y].sort_values("age")
            if len(tmp):
                plt.plot(tmp["age"], tmp["qx"], linewidth=2.1, linestyle="--", label=f"Forecast {y}")
        plt.yscale("log")
        plt.title(f"Mortality schedules: observed vs projected - {sex} - {mv}")
        plt.xlabel("Age")
        plt.ylabel("q_x, log scale")
        plt.legend(frameon=False, ncol=2)
        plt.tight_layout()
        plt.savefig(FIG / f"fig5_qx_schedules_{mv}_{sex}.png", dpi=220)
        plt.savefig(FIG / f"fig5_qx_schedules_{mv}_{sex}.pdf")
        plt.close()

# 6. fit check for recent fitted years
for mv, obj in models.items():
    fit_qx = obj["fit_qx"]
    if fit_qx.empty:
        continue
    for sex in sorted(fit_qx["sex"].unique()):
        plt.figure(figsize=(10, 6))
        g = fit_qx[fit_qx["sex"] == sex].copy()
        for y in sorted(g["year"].unique())[-3:]:
            tmp = g[g["year"] == y].sort_values("age")
            plt.plot(tmp["age"], tmp["qx_observed"], linewidth=1.8, label=f"Observed {y}")
            plt.plot(tmp["age"], tmp["qx_fitted"], linewidth=1.8, linestyle="--", label=f"Fitted {y}")
        plt.yscale("log")
        plt.title(f"Observed vs fitted mortality schedules - {sex} - {mv}")
        plt.xlabel("Age")
        plt.ylabel("q_x, log scale")
        plt.legend(frameon=False, ncol=2)
        plt.tight_layout()
        plt.savefig(FIG / f"fig6_fit_check_{mv}_{sex}.png", dpi=220)
        plt.savefig(FIG / f"fig6_fit_check_{mv}_{sex}.pdf")
        plt.close()

print(f"Figures written to {FIG.resolve()}")

# 7. WPP qx benchmark: model qx vs WPP qx, selected years
wpp_qx = read_if_exists(OUT / "comparison" / "wpp_qx_benchmark.csv")
qx_wpp_comp = read_if_exists(OUT / "comparison" / "mortality_qx_forecast_vs_wpp.csv")
if not qx_wpp_comp.empty:
    qx_wpp_comp["year"] = pd.to_numeric(qx_wpp_comp["year"], errors="coerce")
    qx_wpp_comp["age"] = pd.to_numeric(qx_wpp_comp["age"], errors="coerce")
    for sex in sorted(qx_wpp_comp["sex"].dropna().unique()):
        for mv in MODEL_VERSIONS:
            g = qx_wpp_comp[(qx_wpp_comp["sex"] == sex) & (qx_wpp_comp["model_version"] == mv)].copy()
            if g.empty:
                continue
            available_years = sorted(g["year"].dropna().astype(int).unique())
            pick_years = [2025, 2030, 2050, 2075]
            pick_years = [y for y in pick_years if y in available_years]
            if not pick_years:
                pick_years = [available_years[0], available_years[len(available_years)//2], available_years[-1]]
            plt.figure(figsize=(10, 6))
            for y in pick_years:
                tmp = g[g["year"] == y].sort_values("age")
                plt.plot(tmp["age"], tmp["qx_model"], linewidth=2, label=f"Model {y}")
                plt.plot(tmp["age"], tmp["qx_benchmark"], linewidth=1.7, linestyle="--", label=f"UN WPP {y}")
            plt.yscale("log")
            plt.title(f"Mortality schedules: model vs UN WPP - {sex} - {mv}")
            plt.xlabel("Age")
            plt.ylabel("q_x, log scale")
            plt.legend(frameon=False, ncol=2)
            plt.tight_layout()
            plt.savefig(FIG / f"fig7_qx_model_vs_wpp_{mv}_{sex}.png", dpi=220)
            plt.savefig(FIG / f"fig7_qx_model_vs_wpp_{mv}_{sex}.pdf")
            plt.close()

            plt.figure(figsize=(10, 6))
            for y in pick_years:
                tmp = g[g["year"] == y].sort_values("age")
                plt.plot(tmp["age"], tmp["qx_ratio"], linewidth=2, label=str(y))
            plt.axhline(1.0, linewidth=1, linestyle=":")
            plt.title(f"q_x ratio: model / UN WPP - {sex} - {mv}")
            plt.xlabel("Age")
            plt.ylabel("q_x ratio")
            plt.legend(frameon=False, title="Year")
            plt.tight_layout()
            plt.savefig(FIG / f"fig8_qx_ratio_model_wpp_{mv}_{sex}.png", dpi=220)
            plt.savefig(FIG / f"fig8_qx_ratio_model_wpp_{mv}_{sex}.pdf")
            plt.close()

# 8. life expectancy comparison with benchmark uncertainty ranges when available
bench_all = normalize_benchmark_sex(read_if_exists(OUT / "comparison" / "life_expectancy_benchmarks_e0.csv"))
comp = read_if_exists(OUT / "comparison" / "life_expectancy_forecast_intervals_all_models.csv")
hist_comp = read_if_exists(OUT / "comparison" / "life_expectancy_history_all_models.csv")
if not bench_all.empty and not comp.empty:
    for c in ["year", "e0", "e0_lower90", "e0_upper90", "e0_lower95", "e0_upper95"]:
        if c in bench_all.columns:
            bench_all[c] = pd.to_numeric(bench_all[c], errors="coerce")
    for sex in sorted(comp["sex"].dropna().unique()):
        plt.figure(figsize=(10, 6))
        gh = hist_comp[(hist_comp["sex"] == sex) & (hist_comp["model_version"] == "all_years")].copy() if not hist_comp.empty else pd.DataFrame()
        if not gh.empty:
            plt.plot(gh["year"], gh["e0_observed"], linewidth=2, label="Observed e0")
        for mv in MODEL_VERSIONS:
            gf = comp[(comp["sex"] == sex) & (comp["model_version"] == mv)].copy()
            if gf.empty:
                continue
            plt.plot(gf["year"], gf["p50"], linewidth=2, label=f"Our model: {mv}")
            plt.fill_between(gf["year"], gf["p025"], gf["p975"], alpha=0.10)
        gb = bench_all[bench_all["sex"] == sex].copy()
        for source, gs in gb.groupby("source"):
            gs = gs.sort_values("year")
            has_95 = (
                "e0_lower95" in gs.columns
                and "e0_upper95" in gs.columns
                and gs[["e0_lower95", "e0_upper95"]].notna().all(axis=1).any()
            )
            has_90 = (
                "e0_lower90" in gs.columns
                and "e0_upper90" in gs.columns
                and gs[["e0_lower90", "e0_upper90"]].notna().all(axis=1).any()
            )
            if has_95:
                band = gs.dropna(subset=["e0_lower95", "e0_upper95"])
                plt.fill_between(band["year"], band["e0_lower95"], band["e0_upper95"], alpha=0.08)
            elif has_90:
                band = gs.dropna(subset=["e0_lower90", "e0_upper90"])
                plt.fill_between(band["year"], band["e0_lower90"], band["e0_upper90"], alpha=0.08)
            if "e0" in gs.columns:
                plt.plot(gs["year"], gs["e0"], linewidth=1.8, linestyle=":", label=f"{source}")
        plt.title(f"Life expectancy at birth: model vs ISTAT/Eurostat/UN - {sex}")
        plt.xlabel("Year")
        plt.ylabel("e0")
        plt.legend(frameon=False, ncol=2)
        plt.tight_layout()
        plt.savefig(FIG / f"fig9_e0_model_vs_all_benchmarks_{sex}.png", dpi=220)
        plt.savefig(FIG / f"fig9_e0_model_vs_all_benchmarks_{sex}.pdf")
        plt.close()
