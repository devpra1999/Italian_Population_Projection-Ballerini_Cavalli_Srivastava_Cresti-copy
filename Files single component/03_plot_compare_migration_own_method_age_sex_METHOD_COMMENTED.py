# =============================================================================
# MIGRATION MODULE - FIGURES, AGE-SEX CHECKS, AND BENCHMARKS
# =============================================================================
# Reading guide
# -------------
# This script reads the migration outputs from 02 and produces two kinds of checks:
#   1. total-flow comparison with ISTAT, Eurostat, and UN;
#   2. age-sex diagnostics to see whether the reconstructed single-age profiles
#      are smooth and whether broad age-band totals are preserved.
# =============================================================================

#!/usr/bin/env python3
"""
03_plot_compare_migration_own_method_age_sex.py

Plot the own-method migration forecast with age-sex reconstruction.

Reads outputs from:
    02_fit_forecast_migration_own_method_age_sex.py

Figures:
1. Total immigration, emigration and net migration vs ISTAT/Eurostat/UN.
2. Single-age diagnostic profiles by origin and sex for a selected year.
3. Broad age-band diagnostic bars by origin and sex for a selected year.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


PROJECT_DIR = Path("/Users/andreaballerini/Downloads/projection_project")
FLOW_ORDER = ["immigration", "emigration", "net_migration"]
FLOW_LABELS = {
    "immigration": "Total immigration",
    "emigration": "Total emigration",
    "net_migration": "Total net migration",
}
SOURCE_COLORS = {
    "Observed input": "#222222",
    "Our model": "#1f77b4",
    "ISTAT": "#ff7f0e",
    "Eurostat": "#2ca02c",
    "UN": "#9467bd",
}
ORIGIN_COLORS = {
    ("Italia", "Maschi"): "#1f77b4",
    ("Italia", "Femmine"): "#17becf",
    ("Paesi esteri", "Maschi"): "#ff7f0e",
    ("Paesi esteri", "Femmine"): "#d62728",
}
SCOPE_NOTE = (
    "Scope: total flows, all citizenships. Age detail is allocated by origin, sex and broad age band, "
    "then expanded to ages 0-100."
)
EMPTY_COMP_COLUMNS = ["source", "flow", "measure", "year", "value", "source_file", "unit_note"]


def display_path(path: Path) -> str:
    """
    Format paths in a readable way for console messages.
    """
    try:
        return str(path.resolve().relative_to(PROJECT_DIR.resolve()))
    except Exception:
        return str(path)


def resolve_input_path(path: str | Path | None) -> Optional[Path]:
    """
    Resolve an input path using the project folder defaults. This avoids repeated long
    absolute paths in the rest of the script.
    """
    if path is None or str(path).strip() == "":
        return None
    path = Path(path)
    return path if path.is_absolute() else PROJECT_DIR / path


def resolve_output_path(path: str | Path) -> Path:
    """
    Resolve and create the output directory where model results and diagnostics will be
    saved.
    """
    path = Path(path)
    return path if path.is_absolute() else PROJECT_DIR / path


def norm_text(x: object) -> str:
    s = str(x).strip().lower()
    for a, b in [("à", "a"), ("è", "e"), ("é", "e"), ("ì", "i"), ("ò", "o"), ("ù", "u")]:
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s)


def normalize_source(x: object) -> str:
    """
    Standardize benchmark source names so labels are consistent in the figures.
    """
    s = norm_text(x)
    if "istat" in s:
        return "ISTAT"
    if "eurostat" in s:
        return "Eurostat"
    if s in {"un", "united nations"} or "nations" in s:
        return "UN"
    return str(x).strip()


def normalize_flow(x: object) -> str:
    """
    Standardize migration flow labels such as immigration, emigration, and net migration.
    """
    s = norm_text(x)
    if s in {"imm", "immigration"} or "immigrati" in s:
        return "immigration"
    if s in {"emi", "emigration"} or "emigrati" in s:
        return "emigration"
    if s == "net" or "saldo" in s or "net" in s:
        return "net_migration"
    return s


def normalize_measure(x: object) -> str:
    """
    Standardize benchmark measure labels such as median, lower interval, and upper interval.
    """
    s = norm_text(x)
    if "med" in s or "meidan" in s:
        return "median"
    if ("low" in s or "lower" in s or "inferiore" in s) and "95" in s:
        return "lower_95"
    if ("upp" in s or "high" in s or "higher" in s or "superiore" in s) and "95" in s:
        return "upper_95"
    if ("low" in s or "lower" in s or "inferiore" in s) and "90" in s:
        return "lower_90"
    if ("upp" in s or "high" in s or "higher" in s or "superiore" in s) and "90" in s:
        return "upper_90"
    if "lower x" in s:
        return "lower_x"
    if "higher x" in s or "upper x" in s:
        return "upper_x"
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def empty_comparisons() -> pd.DataFrame:
    """
    Create an empty comparison table with the expected columns when no benchmark file is
    available.
    """
    return pd.DataFrame(columns=EMPTY_COMP_COLUMNS)


def maybe_scale_to_persons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect whether benchmark values are stored in thousands and convert them to persons when
    needed.
    """
    out = df.copy()
    med_abs = out["value"].abs().median(skipna=True)
    if pd.notna(med_abs) and med_abs < 5000:
        out["value"] = out["value"] * 1000.0
        out["unit_note"] = "rescaled_from_thousands_to_persons"
    else:
        out["unit_note"] = "persons"
    return out


def read_comparison_excel(path: Optional[Path]) -> pd.DataFrame:
    """
    Read benchmark migration data from Excel and standardize it for plotting.
    """
    if path is None or not path.exists():
        return empty_comparisons()
    df = pd.read_excel(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"source", "emi/imm/net", "measure", "year", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{display_path(path)} missing columns {missing}. Found: {list(df.columns)}")
    out = df.rename(columns={"emi/imm/net": "flow"})[["source", "flow", "measure", "year", "value"]].copy()
    out["source"] = out["source"].map(normalize_source)
    out["flow"] = out["flow"].map(normalize_flow)
    out["measure"] = out["measure"].map(normalize_measure)
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["year", "value"]).copy()
    out["year"] = out["year"].astype(int)
    out["source_file"] = path.name
    pieces = [maybe_scale_to_persons(g) for _, g in out.groupby(["source", "flow"], dropna=False)]
    if not pieces:
        return empty_comparisons()
    return pd.concat(pieces, ignore_index=True).sort_values(["source", "flow", "measure", "year"])


def derive_net_from_components(comp: pd.DataFrame) -> pd.DataFrame:
    """
    Derive net migration from immigration and emigration when a benchmark source provides
    components but not net migration directly.
    """
    if comp.empty:
        return comp
    out = comp.copy()
    rows = []
    candidates = out[out["flow"].isin(["immigration", "emigration"])].copy()
    for source, source_group in candidates.groupby("source"):
        wide = source_group.pivot_table(index="year", columns=["flow", "measure"], values="value", aggfunc="mean").sort_index()

        def has(flow: str, measure: str) -> bool:
            return (flow, measure) in wide.columns

        specs = []
        if has("immigration", "median") and has("emigration", "median"):
            specs.append(("median", wide[("immigration", "median")] - wide[("emigration", "median")]))
        if has("immigration", "lower_90") and has("emigration", "upper_90"):
            specs.append(("lower_90", wide[("immigration", "lower_90")] - wide[("emigration", "upper_90")]))
        if has("immigration", "upper_90") and has("emigration", "lower_90"):
            specs.append(("upper_90", wide[("immigration", "upper_90")] - wide[("emigration", "lower_90")]))
        if has("immigration", "lower_95") and has("emigration", "upper_95"):
            specs.append(("lower_95", wide[("immigration", "lower_95")] - wide[("emigration", "upper_95")]))
        if has("immigration", "upper_95") and has("emigration", "lower_95"):
            specs.append(("upper_95", wide[("immigration", "upper_95")] - wide[("emigration", "lower_95")]))

        for measure, values in specs:
            tmp = values.dropna().reset_index()
            tmp.columns = ["year", "value"]
            tmp["source"] = source
            tmp["flow"] = "net_migration"
            tmp["measure"] = measure
            tmp["source_file"] = "derived_from_immigration_emigration_components"
            tmp["unit_note"] = "persons"
            rows.append(tmp[EMPTY_COMP_COLUMNS])
    if rows:
        out = pd.concat([out, *rows], ignore_index=True)
    return out.drop_duplicates(["source", "flow", "measure", "year"], keep="first").sort_values(
        ["source", "flow", "measure", "year"]
    )


def source_coverage(comp: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize which benchmark sources and flows are available in the comparison file.
    """
    if comp.empty:
        return pd.DataFrame(columns=["source", "flow", "measure", "first_year", "last_year", "n_years"])
    return (
        comp.groupby(["source", "flow", "measure"], as_index=False)
        .agg(first_year=("year", "min"), last_year=("year", "max"), n_years=("year", "nunique"))
        .sort_values(["source", "flow", "measure"])
    )


def benchmark_vs_model(intervals: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    """
    Merge benchmark and model outputs so differences can be inspected numerically as well as
    visually.
    """
    if comp.empty:
        return pd.DataFrame()
    model = intervals[["year", "flow", "p50"]].rename(columns={"p50": "model_p50"})
    out = comp.merge(model, on=["year", "flow"], how="left")
    out["difference_vs_model_p50"] = out["value"] - out["model_p50"]
    out.insert(2, "citizenship_scope", "total_all_citizenships")
    return out


def interval_columns(wide: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    """
    Detect which lower and upper interval columns are available for a plotted series.
    """
    for low, high in [("lower_90", "upper_90"), ("lower_95", "upper_95"), ("lower_x", "upper_x")]:
        if low in wide.columns and high in wide.columns:
            return low, high
    return None, None


def save_total_plot(
    observed: pd.DataFrame,
    intervals: pd.DataFrame,
    comp: pd.DataFrame,
    flow: str,
    plots_dir: Path,
    end_year: int,
) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 7.0))
    model = intervals[(intervals["flow"].eq(flow)) & (intervals["year"].le(end_year))].sort_values("year")
    c = comp[(comp["flow"].eq(flow)) & (comp["year"].le(end_year))].copy()

    ax.plot(
        observed["year"],
        observed[flow],
        color=SOURCE_COLORS["Observed input"],
        linewidth=1.8,
        marker="o",
        markersize=3.8,
        label="Observed total input",
    )
    ax.fill_between(model["year"], model["p05"], model["p95"], color=SOURCE_COLORS["Our model"], alpha=0.12, linewidth=0, label="Our model 90% interval")
    ax.fill_between(model["year"], model["p10"], model["p90"], color=SOURCE_COLORS["Our model"], alpha=0.20, linewidth=0, label="Our model 80% interval")
    ax.plot(model["year"], model["p50"], color=SOURCE_COLORS["Our model"], linewidth=2.8, label="Our model median")

    for source, g in c.groupby("source"):
        wide = g.pivot_table(index="year", columns="measure", values="value", aggfunc="mean").sort_index()
        color = SOURCE_COLORS.get(source, "#777777")
        low, high = interval_columns(wide)
        if low and high:
            ax.fill_between(wide.index, wide[low], wide[high], color=color, alpha=0.12, linewidth=0, label=f"{source} interval")
        if "median" in wide.columns:
            ax.plot(wide.index, wide["median"], color=color, linestyle="--", linewidth=2.0, label=f"{source} median")

    if flow == "net_migration":
        ax.axhline(0, color="#777777", linewidth=0.8)
    xmin = min(int(observed["year"].min()), int(model["year"].min()))
    ax.set_xlim(xmin, end_year)
    fig.suptitle(f"{FLOW_LABELS[flow]}: own model vs benchmarks", fontsize=18, weight="bold", y=0.985)
    fig.text(0.5, 0.94, SCOPE_NOTE, ha="center", va="center", fontsize=9.5, color="#555555")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Persons", fontsize=12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(True, color="#dddddd", linewidth=0.8)
    ax.legend(loc="best", frameon=True, fontsize=9.5)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.91])

    stems = [
        f"own_method_age_sex_total_{flow}_vs_benchmarks",
        f"fig9_migration_own_method_age_sex_total_{flow}_vs_all_benchmarks",
    ]
    if flow == "net_migration":
        stems.append("own_method_age_sex_total_net_vs_benchmarks")
    for stem in stems:
        fig.savefig(plots_dir / f"{stem}.png", dpi=220)
        fig.savefig(plots_dir / f"{stem}.pdf")
    plt.close(fig)


def save_single_age_profile(single_age: pd.DataFrame, flow: str, year: int, plots_dir: Path) -> None:
    """
    Plot the reconstructed single-age profile to check that age allocation is smooth and
    plausible.
    """
    sub = single_age[(single_age["flow"].eq(flow)) & (single_age["year"].eq(year))].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(12.0, 6.8))
    for (origin, sex), g in sub.groupby(["origin", "sex"]):
        label = f"{origin}, {sex}"
        ax.plot(
            g.sort_values("age")["age"],
            g.sort_values("age")["count"],
            linewidth=2.2,
            color=ORIGIN_COLORS.get((origin, sex)),
            label=label,
        )
    if flow == "net_migration":
        ax.axhline(0, color="#777777", linewidth=0.9)
    fig.suptitle(f"{FLOW_LABELS[flow]} single-age profile, {year}", fontsize=17, weight="bold", y=0.985)
    fig.text(
        0.5,
        0.94,
        "Origin labels are from the migration inputs; population structure uses italiani.csv or stranieri.csv.",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#555555",
    )
    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Persons", fontsize=12)
    ax.set_xlim(0, 100)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(True, color="#dddddd", linewidth=0.8)
    ax.legend(loc="best", frameon=True, fontsize=10)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.91])
    stem = f"migration_single_age_profile_{flow}_{year}"
    fig.savefig(plots_dir / f"{stem}.png", dpi=220)
    fig.savefig(plots_dir / f"{stem}.pdf")
    plt.close(fig)


def save_age_band_bars(age_bands: pd.DataFrame, flow: str, year: int, plots_dir: Path) -> None:
    """
    Plot broad age-band counts to check that the reconstructed distribution remains
    consistent with the input.
    """
    sub = age_bands[(age_bands["flow"].eq(flow)) & (age_bands["year"].eq(year))].copy()
    if sub.empty:
        return
    sub["component"] = sub["origin"] + ", " + sub["sex"]
    sub = sub.sort_values(["age_min", "origin", "sex"])
    bands = list(dict.fromkeys(sub["age_band"].tolist()))
    components = list(dict.fromkeys(sub["component"].tolist()))
    x = np.arange(len(bands))
    width = 0.18
    offsets = np.linspace(-width * 1.5, width * 1.5, len(components))

    fig, ax = plt.subplots(figsize=(11.8, 6.8))
    for offset, component in zip(offsets, components):
        g = sub[sub["component"].eq(component)].set_index("age_band").reindex(bands)
        origin, sex = component.split(", ")
        ax.bar(x + offset, g["count"], width=width, color=ORIGIN_COLORS.get((origin, sex)), label=component)
    if flow == "net_migration":
        ax.axhline(0, color="#777777", linewidth=0.9)
    ax.set_title(f"{FLOW_LABELS[flow]} broad age bands, {year}", fontsize=17, weight="bold")
    ax.set_xlabel("Broad age band", fontsize=12)
    ax.set_ylabel("Persons", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.8)
    ax.legend(loc="best", frameon=True, fontsize=10)
    fig.tight_layout()
    stem = f"migration_age_band_profile_{flow}_{year}"
    fig.savefig(plots_dir / f"{stem}.png", dpi=220)
    fig.savefig(plots_dir / f"{stem}.pdf")
    plt.close(fig)


def main() -> None:
    """
    Run the complete script from inputs to outputs.
    """
    parser = argparse.ArgumentParser(description="Plot own-method age-sex migration forecast against benchmarks.")
    parser.add_argument("--output-dir", default="output/migration/own_method_age_sex")
    parser.add_argument("--comparison-extra", default="migration_altri.xlsx")
    parser.add_argument("--end-year", type=int, default=2075)
    parser.add_argument("--profile-year", type=int, default=None)
    args = parser.parse_args()

    outdir = resolve_output_path(args.output_dir)
    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    observed_path = outdir / "migration_observed_totals_by_year.csv"
    intervals_path = outdir / "migration_own_method_age_sex_forecast_intervals.csv"
    single_age_path = outdir / "migration_single_age_forecast_mean.csv"
    age_bands_path = outdir / "migration_age_band_forecast_mean.csv"
    for path in [observed_path, intervals_path, single_age_path, age_bands_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {display_path(path)}. Run 02_fit_forecast_migration_own_method_age_sex.py first.")

    observed = pd.read_csv(observed_path)
    intervals = pd.read_csv(intervals_path)
    single_age = pd.read_csv(single_age_path)
    age_bands = pd.read_csv(age_bands_path)
    profile_year = args.profile_year if args.profile_year is not None else int(single_age["year"].min())

    comp = derive_net_from_components(read_comparison_excel(resolve_input_path(args.comparison_extra)))
    if comp.empty:
        raise FileNotFoundError("No ISTAT/Eurostat/UN benchmark data found. Expected migration_altri.xlsx by default.")

    coverage = source_coverage(comp)
    coverage.insert(2, "citizenship_scope", "total_all_citizenships")
    coverage.to_csv(outdir / "migration_own_method_age_sex_comparison_source_coverage.csv", index=False)
    comp_out = comp.copy()
    comp_out.insert(2, "citizenship_scope", "total_all_citizenships")
    comp_out.to_csv(outdir / "migration_own_method_age_sex_comparison_cleaned_all_sources.csv", index=False)
    benchmark_vs_model(intervals, comp).to_csv(outdir / "migration_own_method_age_sex_vs_benchmarks_all_sources.csv", index=False)

    for flow in FLOW_ORDER:
        save_total_plot(observed, intervals, comp, flow, plots_dir, args.end_year)
        save_single_age_profile(single_age, flow, profile_year, plots_dir)
        save_age_band_bars(age_bands, flow, profile_year, plots_dir)

    print(f"Saved own-method age-sex migration comparison outputs to: {display_path(outdir)}")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
