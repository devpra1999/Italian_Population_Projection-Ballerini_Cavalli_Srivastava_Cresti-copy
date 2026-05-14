# =============================================================================
# FERTILITY MODULE - FIGURES AND COMPARISONS
# =============================================================================
# Reading guide
# -------------
# This script does not re-estimate the fertility model. It reads the CSV outputs
# produced by the fertility forecasting script and turns them into diagnostic and
# comparison figures. Keeping figures separate from estimation makes the pipeline
# easier to check: if the model changes, run 02; if only the graph style changes,
# run 03.
# =============================================================================

"""
03_plot_fertility_forecast.py

Presentation-ready figures in Python / Spyder.
Reads outputs from 02_fit_forecast_fertility_schmertmann.py and writes PNG/PDF figures.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path("/Users/andreaballerini/Downloads/projection_project")
OUT = PROJECT_DIR / "output" / "fertility"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

param_hist = pd.read_csv(OUT / "schmertmann_parameter_history.csv")
param_fc = pd.read_csv(OUT / "schmertmann_parameter_forecast_mean.csv")
tfr_sims = pd.read_csv(OUT / "tfr_forecast_simulations.csv")
tfr_mean = pd.read_csv(OUT / "tfr_forecast_mean.csv")
sched = pd.read_csv(OUT / "fertility_history_plus_forecast_mean.csv")
fut_sched = pd.read_csv(OUT / "fertility_forecast_schedule_mean.csv")


#************************************************************************


# External benchmark TFR series: ISTAT / UN / Eurostat
bench = pd.read_excel("/Users/andreaballerini/Downloads/tfr altri.xlsx")

# First two columns are assumed to be source and statistic
bench = bench.rename(columns={bench.columns[0]: "source", bench.columns[1]: "stat"})
bench["source"] = bench["source"].ffill()

# Keep only year columns
year_cols = [c for c in bench.columns if str(c).isdigit()]
bench = bench[["source", "stat"] + year_cols].copy()

# Long format
bench = bench.melt(
    id_vars=["source", "stat"],
    value_vars=year_cols,
    var_name="year",
    value_name="tfr"
)

bench["year"] = pd.to_numeric(bench["year"], errors="coerce")
bench["tfr"] = pd.to_numeric(bench["tfr"], errors="coerce")

# Clean labels
bench["source"] = (
    bench["source"].astype(str).str.strip()
    .replace({
        "istat": "ISTAT",
        "Istat": "ISTAT",
        "un": "UN",
        "Un": "UN",
        "eurostat": "Eurostat"
    })
)

bench["stat"] = (
    bench["stat"].astype(str).str.strip()
    .replace({
        "Lower at 90%": "lower90",
        "Median": "median",
        "Higher at 90%": "upper90",
        "Lower 90": "lower90",
        "Upper 90": "upper90"
    })
)

bench = bench.dropna(subset=["year", "tfr"])

#************************************************************************
#Nuovo da testare sempre
# ============================================================
# Approximate total TFR from Italian and foreign projected TFR
# ============================================================

# Current observed share of Italian women among total women
ratio_fixed = 0.88947

# Forecasted share of Italian women among total women
ratio_forecast = pd.DataFrame({
    "year": list(range(2025, 2076)),
    "share_italian_women": [
        0.86611208, 0.86285159, 0.85960337, 0.85636737, 0.85314356,
        0.84993189, 0.84673231, 0.84354477, 0.84036923, 0.83720564,
        0.83405397, 0.83091416, 0.82778616, 0.82466995, 0.82156546,
        0.81847267, 0.81539151, 0.81232195, 0.80926395, 0.80621746,
        0.80318244, 0.80015885, 0.79714664, 0.79414576, 0.79115619,
        0.78817787, 0.78521076, 0.78225482, 0.77931,    0.77637628,
        0.7734536,  0.77054192, 0.7676412,  0.7647514,  0.76187248,
        0.75900439, 0.75614711, 0.75330058, 0.75046477, 0.74763963,
        0.74482512, 0.74202122, 0.73922787, 0.73644503, 0.73367267,
        0.73091074, 0.72815922, 0.72541805, 0.7226872,  0.71996663,
        0.7172563
    ]
})

# Mean TFR by citizenship
tfr_mean_wide = (
    tfr_mean.pivot(index="year", columns="citizenship", values="tfr")
    .reset_index()
)

# Check expected groups
needed_groups = {"italiani", "stranieri"}
missing_groups = needed_groups - set(tfr_mean_wide.columns)
if missing_groups:
    raise ValueError(f"Missing citizenship groups in tfr_mean: {missing_groups}")

# Approximation 1: fixed ratio
tfr_total_fixed = tfr_mean_wide[["year", "italiani", "stranieri"]].copy()
tfr_total_fixed["tfr_total_fixed_ratio"] = (
    ratio_fixed * tfr_total_fixed["italiani"]
    + (1 - ratio_fixed) * tfr_total_fixed["stranieri"]
)

# Approximation 2: forecasted ratio
tfr_total_dynamic = tfr_mean_wide.merge(ratio_forecast, on="year", how="left")
tfr_total_dynamic["tfr_total_dynamic_ratio"] = (
    tfr_total_dynamic["share_italian_women"] * tfr_total_dynamic["italiani"]
    + (1 - tfr_total_dynamic["share_italian_women"]) * tfr_total_dynamic["stranieri"]
)

# Also build total TFR simulation bands using the dynamic ratio
tfr_sims_wide = (
    tfr_sims.pivot(index=["sim", "year"], columns="citizenship", values="tfr")
    .reset_index()
)

missing_groups_sim = needed_groups - set(tfr_sims_wide.columns)
if missing_groups_sim:
    raise ValueError(f"Missing citizenship groups in tfr_sims: {missing_groups_sim}")

tfr_sims_total = tfr_sims_wide.merge(ratio_forecast, on="year", how="left")
tfr_sims_total["tfr_total_dynamic_ratio"] = (
    tfr_sims_total["share_italian_women"] * tfr_sims_total["italiani"]
    + (1 - tfr_sims_total["share_italian_women"]) * tfr_sims_total["stranieri"]
)

tfr_total_bands = (
    tfr_sims_total.groupby("year")["tfr_total_dynamic_ratio"]
    .quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    .unstack()
    .reset_index()
    .rename(columns={0.1: "p10", 0.25: "p25", 0.5: "p50", 0.75: "p75", 0.9: "p90"})
)

#************************************************************************


#************************************************************************
#Nuovo da testare
for cit, gsim in tfr_sims.groupby("citizenship"):
    gmean = tfr_mean[tfr_mean["citizenship"] == cit].copy()

    bands = (
        gsim.groupby("year")["tfr"]
        .quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        .unstack()
        .reset_index()
        .rename(columns={0.1: "p10", 0.25: "p25", 0.5: "p50", 0.75: "p75", 0.9: "p90"})
    )

    plt.figure(figsize=(10, 6))

    # Your model
    plt.fill_between(bands["year"], bands["p10"], bands["p90"], alpha=0.20, label="My model 10-90%")
    plt.fill_between(bands["year"], bands["p25"], bands["p75"], alpha=0.30, label="My model 25-75%")
    plt.plot(gmean["year"], gmean["tfr"], linewidth=2, label="My model mean")
    plt.plot(bands["year"], bands["p50"], linewidth=1.5, linestyle="--", label="My model median")

    # ISTAT
    istat = bench[bench["source"] == "ISTAT"].copy()
    if not istat.empty:
        istat_w = istat.pivot(index="year", columns="stat", values="tfr").reset_index()
        if {"lower90", "upper90"}.issubset(istat_w.columns):
            plt.fill_between(
                istat_w["year"], istat_w["lower90"], istat_w["upper90"],
                alpha=0.12, label="ISTAT 90% interval"
            )
        if "median" in istat_w.columns:
            plt.plot(
                istat_w["year"], istat_w["median"],
                linewidth=2, linestyle=":", label="ISTAT median"
            )

    # UN
    un = bench[bench["source"] == "UN"].copy()
    if not un.empty:
        un_w = un.pivot(index="year", columns="stat", values="tfr").reset_index()
        if {"lower90", "upper90"}.issubset(un_w.columns):
            plt.fill_between(
                un_w["year"], un_w["lower90"], un_w["upper90"],
                alpha=0.10, label="UN 90% interval"
            )
        if "median" in un_w.columns:
            plt.plot(
                un_w["year"], un_w["median"],
                linewidth=2, linestyle="-.", label="UN median"
            )

    # Eurostat
    euro = bench[bench["source"] == "Eurostat"].copy()
    if not euro.empty:
        euro_w = euro.pivot(index="year", columns="stat", values="tfr").reset_index()
        if "median" in euro_w.columns:
            plt.plot(
                euro_w["year"], euro_w["median"],
                linewidth=2, linestyle="solid", label="Eurostat median"
            )

    plt.title(f"TFR forecast comparison - {cit}")
    plt.xlabel("Year")
    plt.ylabel("TFR")
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(FIG / f"fig1_tfr_comparison_{cit}.png", dpi=220)
    plt.savefig(FIG / f"fig1_tfr_comparison_{cit}.pdf")
    plt.close()


#************************************************************************


for cit, gsim in tfr_sims.groupby("citizenship"):
    gmean = tfr_mean[tfr_mean["citizenship"] == cit].copy()
    bands = (
        gsim.groupby("year")["tfr"]
        .quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        .unstack()
        .reset_index()
        .rename(columns={0.1:"p10",0.25:"p25",0.5:"p50",0.75:"p75",0.9:"p90"})
    )
    plt.figure(figsize=(9,5))
    plt.fill_between(bands["year"], bands["p10"], bands["p90"], alpha=0.20, label="10-90% interval")
    plt.fill_between(bands["year"], bands["p25"], bands["p75"], alpha=0.30, label="25-75% interval")
    plt.plot(gmean["year"], gmean["tfr"], linewidth=2, label="Mean forecast")
    plt.plot(bands["year"], bands["p50"], linewidth=1.5, linestyle="--", label="Median simulation")
    plt.title(f"TFR forecast with uncertainty - {cit}")
    plt.xlabel("Year")
    plt.ylabel("TFR")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIG / f"fig1_tfr_fanchart_{cit}.png", dpi=220)
    plt.savefig(FIG / f"fig1_tfr_fanchart_{cit}.pdf")
    plt.close()

########################################################################

for cit in sorted(param_hist["citizenship"].unique()):
    gh = param_hist[param_hist["citizenship"] == cit].copy()
    gf = param_fc[param_fc["citizenship"] == cit].copy()
    plt.figure(figsize=(10,6))
    for col in ["alpha","P","H"]:
        plt.plot(gh["year"], gh[col], linewidth=2, label=f"{col} observed")
        plt.plot(gf["year"], gf[col], linewidth=2, linestyle="--", label=f"{col} forecast")
    plt.title(f"Schmertmann parameters over time - {cit}")
    plt.xlabel("Year")
    plt.ylabel("Age")
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(FIG / f"fig2_parameters_{cit}.png", dpi=220)
    plt.savefig(FIG / f"fig2_parameters_{cit}.pdf")
    plt.close()

pick_years = []
if len(fut_sched):
    years = sorted(fut_sched["year"].unique())
    pick_years = [years[0], years[min(9, len(years)-1)], years[-1]]

for cit in sorted(sched["citizenship"].unique()):
    plt.figure(figsize=(10,6))
    gh = sched[(sched["citizenship"] == cit) & (sched["source"] == "observed")]
    if len(gh):
        for y in sorted(gh["year"].unique())[-3:]:
            tmp = gh[gh["year"] == y]
            plt.plot(tmp["age_mother"], tmp["asfr"], linewidth=1.8, label=f"Observed {y}")
    gf = fut_sched[fut_sched["citizenship"] == cit]
    for y in pick_years:
        tmp = gf[gf["year"] == y]
        if len(tmp):
            plt.plot(tmp["age_mother"], tmp["asfr"], linewidth=2.2, linestyle="--", label=f"Forecast {y}")
    plt.title(f"ASFR schedules: observed vs projected - {cit}")
    plt.xlabel("Age of mother")
    plt.ylabel("ASFR")
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(FIG / f"fig3_asfr_schedules_{cit}.png", dpi=220)
    plt.savefig(FIG / f"fig3_asfr_schedules_{cit}.pdf")
    plt.close()

for cit, g in param_hist.groupby("citizenship"):
    plt.figure(figsize=(9,4.5))
    plt.plot(g["year"], g["rmse_shape"], linewidth=2)
    plt.title(f"Schmertmann fit RMSE by year - {cit}")
    plt.xlabel("Year")
    plt.ylabel("RMSE of standardized schedule")
    plt.tight_layout()
    plt.savefig(FIG / f"fig4_fit_rmse_{cit}.png", dpi=220)
    plt.savefig(FIG / f"fig4_fit_rmse_{cit}.pdf")
    plt.close()

print(f"Figures written to {FIG.resolve()}")



##Nuovo da provare


# ============================================================
# Figure: total TFR comparison with ISTAT / UN / Eurostat
# ============================================================

plt.figure(figsize=(10, 6))

# My model uncertainty band based on dynamic ratio
plt.fill_between(
    tfr_total_bands["year"], tfr_total_bands["p10"], tfr_total_bands["p90"],
    alpha=0.18, label="My total TFR 10-90%"
)
plt.fill_between(
    tfr_total_bands["year"], tfr_total_bands["p25"], tfr_total_bands["p75"],
    alpha=0.28, label="My total TFR 25-75%"
)

# My model mean: fixed-ratio approximation
plt.plot(
    tfr_total_fixed["year"], tfr_total_fixed["tfr_total_fixed_ratio"],
    linewidth=2, linestyle="--", label="My total TFR (fixed Italian share)"
)

# My model mean: dynamic-ratio approximation
plt.plot(
    tfr_total_dynamic["year"], tfr_total_dynamic["tfr_total_dynamic_ratio"],
    linewidth=2.5, label="My total TFR (forecasted Italian share)"
)

# Median of simulated total TFR
plt.plot(
    tfr_total_bands["year"], tfr_total_bands["p50"],
    linewidth=1.5, linestyle=":", label="My total TFR median simulation"
)

# ISTAT
istat = bench[bench["source"] == "ISTAT"].copy()
if not istat.empty:
    istat_w = istat.pivot(index="year", columns="stat", values="tfr").reset_index()
    if {"lower90", "upper90"}.issubset(istat_w.columns):
        plt.fill_between(
            istat_w["year"], istat_w["lower90"], istat_w["upper90"],
            alpha=0.10, label="ISTAT 90% interval"
        )
    if "median" in istat_w.columns:
        plt.plot(
            istat_w["year"], istat_w["median"],
            linewidth=2, linestyle="-.", label="ISTAT median"
        )

# UN
un = bench[bench["source"] == "UN"].copy()
if not un.empty:
    un_w = un.pivot(index="year", columns="stat", values="tfr").reset_index()
    if {"lower90", "upper90"}.issubset(un_w.columns):
        plt.fill_between(
            un_w["year"], un_w["lower90"], un_w["upper90"],
            alpha=0.08, label="UN 90% interval"
        )
    if "median" in un_w.columns:
        plt.plot(
            un_w["year"], un_w["median"],
            linewidth=2, linestyle="solid", label="UN median"
        )

# Eurostat
euro = bench[bench["source"] == "Eurostat"].copy()
if not euro.empty:
    euro_w = euro.pivot(index="year", columns="stat", values="tfr").reset_index()
    if "median" in euro_w.columns:
        plt.plot(
            euro_w["year"], euro_w["median"],
            linewidth=2, linestyle=(0, (3, 1, 1, 1)), label="Eurostat median"
        )

plt.title("Total TFR comparison: model vs ISTAT / UN / Eurostat")
plt.xlabel("Year")
plt.ylabel("TFR")
plt.legend(frameon=False, ncol=2)
plt.tight_layout()
plt.savefig(FIG / "fig_total_tfr_comparison.png", dpi=220)
plt.savefig(FIG / "fig_total_tfr_comparison.pdf")
plt.close()
