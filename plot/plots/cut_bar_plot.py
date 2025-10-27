# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from ..utils import PlotMetadata, build_tab10_palette, draw_errorbars, ensure_dir, format_bytes, sort_key


def _resolve_output_dir(system: str, output_dir: str | Path | None) -> Path:
    return ensure_dir(output_dir) if output_dir else ensure_dir(Path("plot") / system)


def generate_cut_bar_plot(
    data: pd.DataFrame,
    *,
    metadata: PlotMetadata,
    collective: str,
    datatype: str,
    std_threshold: float = 0.5,
    output_dir: str | Path | None = None,
) -> Path:
    """
    Render the split bar plot that emphasises small vs large differences.
    ``data`` must contain ``normalized_mean`` and ``normalized_std`` columns.
    """
    if data.empty:
        raise ValueError("No data available for generate_cut_bar_plot.")

    sorted_algos = sorted(data["algo_name"].unique().tolist(), key=sort_key)
    palette = build_tab10_palette(sorted_algos)

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": [1, 3]},
        figsize=(12, 8),
    )

    sns.barplot(
        ax=ax_top,
        data=data,
        x="buffer_size",
        y="normalized_mean",
        hue="algo_name",
        hue_order=sorted_algos,
        palette=palette,
        errorbar=None,
    )
    sns.barplot(
        ax=ax_bot,
        data=data,
        x="buffer_size",
        y="normalized_mean",
        hue="algo_name",
        hue_order=sorted_algos,
        palette=palette,
        errorbar=None,
    )

    if ax_top.get_legend():
        ax_top.get_legend().remove()

    y_min = 1.8
    y_max = min(data["normalized_mean"].max() * 1.1, 10.0)

    draw_errorbars(
        ax_top,
        data,
        sorted_algos,
        std_threshold,
        threshold_mode="relative",
        loc=(y_max - y_min) * 0.1,
        top=True,
        y_min=y_min,
    )
    draw_errorbars(
        ax_bot,
        data,
        sorted_algos,
        std_threshold,
        threshold_mode="relative",
        loc=0.05,
        y_min=y_min,
    )
    ax_bot.set_ylim(0, y_min - 0.05)
    ax_top.set_ylim(y_min, y_max)

    top_limit = ax_top.get_ylim()[1]
    for container in ax_top.containers:
        for bar in container:
            if hasattr(bar, "get_height") and bar.get_height() > top_limit:
                x = bar.get_x() + bar.get_width() / 2.0
                ax_top.scatter(x, top_limit - 0.5, marker="^", color="black", s=100, zorder=4)

    ax_top.spines["bottom"].set_visible(True)
    ax_bot.spines["top"].set_visible(True)
    ax_top.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

    d = 0.005
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    ax_top.grid(True, which="both", linestyle="-", linewidth=0.25, axis="y")
    ax_bot.grid(True, which="both", linestyle="-", linewidth=0.5, axis="y")

    ax_bot.set_xlabel("Buffer Size", fontsize=15)
    ax_bot.set_ylabel("Normalized Mean Execution Time", fontsize=15)
    ax_top.set_ylabel("")

    if metadata.total_nodes == metadata.mpi_tasks:
        title = f"{metadata.system}, {collective.lower()}, {metadata.nnodes} nodes ({datatype})"
    else:
        title = (
            f"{metadata.system}, {collective.lower()}, {metadata.nnodes} nodes "
            f"({datatype}, {metadata.mpi_tasks} tasks)"
        )
    fig.suptitle(title, fontsize=18)

    ax_bot.set_xticks(ax_bot.get_xticks())
    new_labels = [format_bytes(tick.get_text()) for tick in ax_bot.get_xticklabels()]
    ax_bot.set_xticklabels(new_labels)

    handles, labels = ax_bot.get_legend_handles_labels()
    if handles:
        if len(handles) > 10:
            ncols = 3
        elif len(handles) > 5:
            ncols = 2
        else:
            ncols = 1
        ax_bot.legend(handles, labels, ncol=ncols, loc="lower left", fontsize=9)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    target_dir = _resolve_output_dir(metadata.system, output_dir)
    name = f"{collective.lower()}_{metadata.nnodes}_{datatype}_{metadata.timestamp}_barplot_cut.png"
    full_path = target_dir / name
    plt.savefig(full_path, dpi=300)
    plt.close()
    return full_path
