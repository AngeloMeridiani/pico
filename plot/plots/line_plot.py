from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..utils import (
    PlotMetadata,
    build_tab10_palette,
    ensure_dir,
    format_time_units_ns,
    format_bytes,
    sort_key,
)


sns.set_palette("tab10")
sns.set_style("whitegrid")
sns.set_context("paper")


def _resolve_output_dir(system: str, output_dir: str | Path | None) -> Path:
    if output_dir:
        return ensure_dir(output_dir)
    return ensure_dir(Path("plot") / system)


def generate_line_plot(
    data: pd.DataFrame,
    *,
    metadata: PlotMetadata,
    collective: str,
    datatype: str,
    error_col: str | None = "se",
    error_mode: str = "band",
    output_dir: str | Path | None = None,
) -> Path:
    """
    Render the latency line plot (log-log) for a single ``collective`` and
    ``datatype`` pair.  ``data`` must already be filtered to contain only rows
    for the selected pair.
    """
    if data.empty:
        raise ValueError("No data available for generate_line_plot.")

    sorted_algos = sorted(data["algo_name"].unique().tolist(), key=sort_key)
    palette = build_tab10_palette(sorted_algos)
    df = data.sort_values("buffer_size").copy()

    err_series = None
    if error_col == "std" and {"std", "n_iter"}.issubset(df.columns):
        err_series = df["std"]
    elif error_col == "ci" and {"ci_lower", "ci_upper"}.issubset(df.columns):
        err_series = (df["ci_upper"] - df["ci_lower"]) / 2.0
    elif error_col == "se" and "standard_error" in df.columns:
        err_series = df["standard_error"]
    elif error_col == "iqr" and {"iqr"}.issubset(df.columns):
        err_series = df["iqr"] / 2.0
    elif error_col == "percentiles" and {"percentile_10", "percentile_90"}.issubset(df.columns):
        err_series = (df["percentile_90"] - df["percentile_10"]) / 2.0

    plt.figure(figsize=(12, 8))
    ax = sns.lineplot(
        data=df,
        x="buffer_size",
        y="mean",
        hue="algo_name",
        style="algo_name",
        hue_order=sorted_algos,
        palette=palette,
        markers=True,
        markersize=9,
        linewidth=2,
    )

    if err_series is not None:
        df = df.assign(__err=err_series.astype(float))
        if error_mode == "band":
            for algo, group in df.groupby("algo_name"):
                g = group.sort_values("buffer_size")
                x = g["buffer_size"].values
                y = g["mean"].values
                e = g["__err"].values
                ax.fill_between(x, y - e, y + e, alpha=0.15, zorder=2, color=palette.get(algo))
        else:
            for algo, group in df.groupby("algo_name"):
                g = group.sort_values("buffer_size")
                ax.errorbar(
                    g["buffer_size"].values,
                    g["mean"].values,
                    yerr=g["__err"].values,
                    fmt="none",
                    ecolor="black",
                    elinewidth=1,
                    capsize=3,
                    zorder=4,
                )

    plt.xscale("log")
    plt.yscale("log")

    xticks = sorted(pd.unique(df["buffer_size"]).astype(float))
    ax.set_xticks(xticks)
    ax.set_xticklabels([format_bytes(x) for x in xticks])
    ax.yaxis.set_major_formatter(format_time_units_ns)

    if metadata.nnodes == metadata.mpi_tasks:
        title = f"{metadata.system.capitalize()}, {collective.lower().capitalize()}, {metadata.nnodes} nodes"
    else:
        title = (
            f"{metadata.system.capitalize()}, {collective.lower().capitalize()}, "
            f"{metadata.nnodes} nodes ({metadata.mpi_tasks} tasks)"
        )

    plt.title(title, fontsize=18)
    ax.set_xlabel("Message Size", fontsize=15)
    ax.set_ylabel("Execution Time", fontsize=15)

    if len(sorted_algos) > 10:
        ncols = 3
    elif len(sorted_algos) > 5:
        ncols = 2
    else:
        ncols = 1

    handles, labels = ax.get_legend_handles_labels()
    new_labels = [
        " ".join(w.capitalize() for w in label.replace("_", " ").split() if w not in {"over", "ompi", "distance"})
        for label in labels
    ]
    ax.legend(handles, new_labels, fontsize=20, loc="upper left", ncol=ncols, frameon=True)
    plt.tight_layout()

    target_dir = _resolve_output_dir(metadata.system, output_dir)
    suffix = f"{error_col}_lineplot" if error_col else "lineplot"
    name = f"{collective.lower()}_{metadata.nnodes}_{datatype}_{metadata.timestamp}_{suffix}.png"
    full_path = target_dir / name
    plt.savefig(full_path, dpi=300)
    plt.close()
    return full_path
