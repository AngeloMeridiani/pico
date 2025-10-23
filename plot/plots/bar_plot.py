from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..utils import PlotMetadata, build_tab10_palette, draw_errorbars, ensure_dir, format_bytes, sort_key


def _resolve_output_dir(system: str, output_dir: str | Path | None) -> Path:
        return ensure_dir(output_dir) if output_dir else ensure_dir(Path("plot") / system)


def generate_bar_plot(
    data: pd.DataFrame,
    *,
    metadata: PlotMetadata,
    collective: str,
    datatype: str,
    std_threshold: float = 0.15,
    output_dir: str | Path | None = None,
) -> Path:
    """
    Render the normalized bar plot for a specific ``collective`` / ``datatype`` pair.
    The incoming dataframe must already contain ``normalized_mean`` and ``normalized_std``.
    """
    if data.empty:
        raise ValueError("No data available for generate_bar_plot.")

    sorted_algos = sorted(data["algo_name"].unique().tolist(), key=sort_key)

    plt.figure(figsize=(12, 8))
    palette = build_tab10_palette(sorted_algos)
    ax = sns.barplot(
        data=data,
        x="buffer_size",
        y="normalized_mean",
        hue="algo_name",
        hue_order=sorted_algos,
        errorbar=None,
        palette=palette,
    )

    draw_errorbars(ax, data, sorted_algos, std_threshold, threshold_mode="absolute", loc=0.05)

    ax.set_xticks(ax.get_xticks())
    new_labels = []
    for tick in ax.get_xticklabels():
        new_labels.append(format_bytes(tick.get_text()))
    ax.set_xticklabels(new_labels)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        if len(handles) > 10:
            ncols = 3
        elif len(handles) > 5:
            ncols = 2
        else:
            ncols = 1
        new_labels = [
            " ".join(w.capitalize() for w in label.replace("_", " ").split() if w not in {"over", "ompi", "distance"})
            for label in labels
        ]
        ax.legend(handles, new_labels, ncol=ncols, loc="lower left", fontsize=20)

    if metadata.nnodes == metadata.mpi_tasks:
        title = f"{metadata.system.capitalize()}, {collective.lower().capitalize()}, {metadata.nnodes} nodes"
    else:
        title = (
            f"{metadata.system.capitalize()}, {collective.lower().capitalize()}, "
            f"{metadata.nnodes} nodes ({metadata.mpi_tasks} tasks)"
        )
    plt.title(title, fontsize=18)
    plt.xlabel("Message Size", fontsize=15)
    plt.ylabel("Normalized Execution Time", fontsize=15)
    plt.grid(True, which="both", linestyle="-", linewidth=0.5, axis="y")
    plt.tight_layout()

    target_dir = _resolve_output_dir(metadata.system, output_dir)
    name = f"{collective.lower()}_{metadata.nnodes}_{datatype}_{metadata.timestamp}_barplot.png"
    full_path = target_dir / name
    plt.savefig(full_path, dpi=300)
    plt.close()
    return full_path
