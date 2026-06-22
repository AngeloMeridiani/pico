# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from ..utils import ensure_dir, save_figure
from .comparison_heatmap import BIG_FONT_SIZE, SMALL_FONT_SIZE
from .default_vs_best_heatmap import (
    DEFAULT_BASELINE_PATTERN,
    METRICS,
    _build_algo_codes,
    _format_ppn,
    _infer_nodes,
    _node_sort_key,
    _parse_run_entry,
    _text_color_for_rgba,
)


def _format_vector_size_label(num_bytes: int) -> str:
    """
    Render vector sizes with IEC units (B, KiB, MiB).
    Values >= 1 MiB are kept in MiB.
    """
    value = float(num_bytes)
    if value < 1024:
        return f"{int(value)} B"
    if value < 1024**2:
        kib = value / 1024.0
        return f"{int(round(kib))} KiB" if abs(kib - round(kib)) < 1e-9 else f"{kib:.1f} KiB"
    mib = value / (1024.0**2)
    return f"{int(round(mib))} MiB" if abs(mib - round(mib)) < 1e-9 else f"{mib:.1f} MiB"


def _is_no_bine_exclude(exclude: str | None) -> bool:
    return bool(exclude and re.search(r"bine", exclude, flags=re.IGNORECASE))


@dataclass(slots=True)
class DefaultVsBestDualHeatmapConfig:
    system: str
    collective: str
    runs: Iterable[str]
    metric: str = "mean"
    default_pattern: str = DEFAULT_BASELINE_PATTERN
    ppn_left: int = 1
    ppn_right: int = 4
    exclude: str | None = None
    title: str | None = None
    output: str | Path | None = None
    output_format: str = "pdf"


def _compute_matrix_for_ppn(
    cfg: DefaultVsBestDualHeatmapConfig,
    ppn: int,
    collective: str,
    metric: str,
) -> tuple[pd.DataFrame, str]:
    frames: list[pd.DataFrame] = []
    tasks_per_node_values: set[float] = set()

    for raw_entry in cfg.runs:
        timestamp, run_nodes = _parse_run_entry(raw_entry)
        summary_path = Path("results") / cfg.system / timestamp / "aggregated_results_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Summary file {summary_path} not found.")

        df = pd.read_csv(summary_path)
        subset = df[df["collective_type"].str.lower() == collective.lower()].copy()
        if subset.empty:
            continue

        if cfg.exclude:
            subset = subset[~subset["algo_name"].str.contains(cfg.exclude, case=False, na=False)]
        if subset.empty:
            continue

        if "tasks_per_node" not in subset.columns:
            raise RuntimeError(
                f"Run {timestamp} has no tasks_per_node column, cannot select {ppn} tasks-per-node."
            )
        subset["tasks_per_node_value"] = pd.to_numeric(subset["tasks_per_node"], errors="coerce")
        subset = subset[np.isclose(subset["tasks_per_node_value"], float(ppn))]
        if subset.empty:
            continue
        tasks_per_node_values.update(
            subset["tasks_per_node_value"].dropna().astype(float).unique().tolist()
        )

        node_value = run_nodes or _infer_nodes(subset)
        subset["Nodes"] = str(node_value)
        subset["time"] = pd.to_numeric(subset[metric], errors="coerce")
        subset = subset[np.isfinite(subset["time"]) & (subset["time"] > 0)]
        if subset.empty:
            continue

        subset = subset.loc[
            subset.groupby(
                ["buffer_size", "Nodes", "algo_name", "tasks_per_node_value"]
            )["time"].idxmin()
        ]
        frames.append(
            subset[
                ["buffer_size", "Nodes", "algo_name", "time", "tasks_per_node_value"]
            ]
        )

    if not frames:
        raise RuntimeError(f"No matching data found for {collective} at {ppn} tasks-per-node.")

    data = pd.concat(frames, ignore_index=True)
    all_cells = data[["buffer_size", "Nodes"]].drop_duplicates().copy()
    all_cells["buffer_size"] = all_cells["buffer_size"].astype(int)
    all_cells["Nodes"] = all_cells["Nodes"].astype(str)
    records: list[dict[str, object]] = []

    for (buffer_size, nodes, _), group in data.groupby(
        ["buffer_size", "Nodes", "tasks_per_node_value"]
    ):
        default_mask = group["algo_name"].astype(str).str.fullmatch(
            cfg.default_pattern,
            case=False,
            na=False,
        )
        default_rows = group[default_mask]
        if default_rows.empty:
            continue

        default_row = default_rows.loc[default_rows["time"].idxmin()]
        default_time = float(default_row["time"])
        if not np.isfinite(default_time) or default_time <= 0:
            continue

        candidates = group[~default_mask]
        if candidates.empty:
            continue

        best_row = candidates.loc[candidates["time"].idxmin()]
        best_time = float(best_row["time"])
        if not np.isfinite(best_time) or best_time <= 0:
            continue

        records.append(
            {
                "buffer_size": int(buffer_size),
                "Nodes": str(nodes),
                "best_algo": str(best_row["algo_name"]),
                "ratio": best_time / default_time,
            }
        )

    if not records:
        raise RuntimeError(
            f"No cells could be computed for {collective} at {ppn} tasks-per-node."
        )

    matrix = pd.DataFrame.from_records(records)
    matrix = all_cells.merge(matrix, on=["buffer_size", "Nodes"], how="left")
    return matrix, _format_ppn(tasks_per_node_values)


def _panel_frames(
    matrix: pd.DataFrame,
    buffers: list[int],
    nodes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratio_data = matrix.pivot(index="buffer_size", columns="Nodes", values="ratio")
    ratio_data = ratio_data.reindex(index=buffers, columns=nodes)

    algo_data = matrix.pivot(index="buffer_size", columns="Nodes", values="best_algo")
    algo_data = algo_data.reindex(index=buffers, columns=nodes)
    return ratio_data, algo_data


def _annotate_cells(
    ax,
    ratio_data: pd.DataFrame,
    algo_data: pd.DataFrame,
    algo_to_idx: dict[str, int],
    algo_codes: dict[str, str],
    colors: list[tuple[float, float, float, float]],
) -> None:
    for i in range(ratio_data.shape[0]):
        for j in range(ratio_data.shape[1]):
            algo = algo_data.iloc[i, j]
            ratio = ratio_data.iloc[i, j]
            if pd.isna(algo) or pd.isna(ratio):
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    fontsize=SMALL_FONT_SIZE - 2,
                    color="black",
                )
                continue

            algo = str(algo)
            color = colors[algo_to_idx[algo]]
            text_color = _text_color_for_rgba(color)
            ax.text(
                j + 0.5,
                i + 0.36,
                algo_codes[algo],
                ha="center",
                va="center",
                fontsize=BIG_FONT_SIZE - 2,
                weight="bold",
                color=text_color,
            )
            ax.text(
                j + 0.5,
                i + 0.74,
                f"{float(ratio):.2f}",
                ha="center",
                va="center",
                fontsize=SMALL_FONT_SIZE - 2,
                color=text_color,
            )


def generate_default_vs_best_dual_heatmap(cfg: DefaultVsBestDualHeatmapConfig) -> Path:
    metric = cfg.metric.lower()
    if metric not in METRICS:
        raise ValueError(f"Unsupported metric '{cfg.metric}'. Expected one of {METRICS}.")
    try:
        re.compile(cfg.default_pattern)
    except re.error as exc:
        raise ValueError(f"Invalid --default-pattern regex: {exc}") from exc

    collective = cfg.collective.upper()
    left_matrix, left_ppn_label = _compute_matrix_for_ppn(cfg, cfg.ppn_left, collective, metric)
    right_matrix, right_ppn_label = _compute_matrix_for_ppn(cfg, cfg.ppn_right, collective, metric)

    ordered_nodes = sorted(
        set(left_matrix["Nodes"].unique().tolist()) | set(right_matrix["Nodes"].unique().tolist()),
        key=_node_sort_key,
    )
    ordered_buffers = sorted(
        set(left_matrix["buffer_size"].unique().tolist())
        | set(right_matrix["buffer_size"].unique().tolist())
    )

    left_ratio, left_algo = _panel_frames(left_matrix, ordered_buffers, ordered_nodes)
    right_ratio, right_algo = _panel_frames(right_matrix, ordered_buffers, ordered_nodes)

    unique_algos = list(
        dict.fromkeys(
            left_matrix["best_algo"].dropna().astype(str).drop_duplicates().tolist()
            + right_matrix["best_algo"].dropna().astype(str).drop_duplicates().tolist()
        )
    )
    algo_to_idx = {algo: idx for idx, algo in enumerate(unique_algos)}
    algo_codes = _build_algo_codes(unique_algos)

    tab10 = plt.get_cmap("tab10")
    colors = [tab10(i % 10) for i in range(len(unique_algos))]
    cmap = ListedColormap(colors)
    cmap.set_bad(color="white")

    n_rows = len(ordered_buffers)
    fig_height = max(8.5, min(28.0, 0.40 * n_rows + 4.0))
    fig, axes = plt.subplots(1, 2, figsize=(17.0, fig_height), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    panels = [
        (axes[0], left_ratio, left_algo, f"{left_ppn_label} PPN"),
        (axes[1], right_ratio, right_algo, f"{right_ppn_label} PPN"),
    ]

    max_visible_y_labels = 12
    y_tick_stride = max(1, int(np.ceil(n_rows / max_visible_y_labels)))
    y_tick_indices = np.arange(0, n_rows, y_tick_stride, dtype=int).tolist()
    if y_tick_indices and y_tick_indices[-1] != n_rows - 1:
        y_tick_indices.append(n_rows - 1)
    y_tick_indices = np.array(sorted(set(y_tick_indices)), dtype=int)
    y_tick_positions = y_tick_indices + 0.5
    y_tick_labels = [_format_vector_size_label(int(ordered_buffers[i])) for i in y_tick_indices]
    y_label_fontsize = max(7, SMALL_FONT_SIZE - 4)

    for idx, (ax, ratio_data, algo_data, panel_title) in enumerate(panels):
        numeric_data = algo_data.apply(lambda col: col.map(algo_to_idx))
        sns.heatmap(
            numeric_data,
            cmap=cmap,
            vmin=-0.5,
            vmax=(len(unique_algos) - 1) + 0.5,
            cbar=False,
            annot=False,
            ax=ax,
        )

        _annotate_cells(ax, ratio_data, algo_data, algo_to_idx, algo_codes, colors)

        ax.set_xticks(np.arange(len(ordered_nodes)) + 0.5)
        ax.set_xticklabels(ordered_nodes, fontsize=SMALL_FONT_SIZE - 1)
        ax.set_xlabel("# Nodes", fontsize=SMALL_FONT_SIZE)
        ax.set_title(panel_title, fontsize=BIG_FONT_SIZE)

        if idx == 0:
            ax.set_ylabel("Vector Size", fontsize=BIG_FONT_SIZE)
        else:
            ax.set_ylabel("")

    # Apply y ticks/labels after both heatmaps are drawn (sharey can override them otherwise).
    axes[0].set_yticks(y_tick_positions)
    axes[0].set_yticklabels(
        y_tick_labels,
        fontsize=y_label_fontsize,
        rotation=0,
        va="center",
        ha="right",
    )
    axes[0].tick_params(axis="y", labelleft=True)
    axes[1].set_yticks(y_tick_positions)
    # Do not set empty ticklabels on shared axis (it can blank both panels).
    axes[1].tick_params(axis="y", labelleft=False, left=False)

    title = cfg.title or (
        f"{cfg.system.capitalize()}, {collective.lower().capitalize()}"
    )
    if _is_no_bine_exclude(cfg.exclude) and "(no Bine)" not in title:
        title = f"{title} (no Bine)"
    fig.suptitle(title, fontsize=BIG_FONT_SIZE + 1, y=0.99)

    legend_handles = [
        Patch(
            facecolor=colors[algo_to_idx[algo]],
            edgecolor="black",
            label=f"{algo_codes[algo]}: {algo}",
        )
        for algo in unique_algos
    ]
    legend_ncol = min(4, max(1, len(legend_handles)))
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=legend_ncol,
        frameon=True,
        fontsize=SMALL_FONT_SIZE - 2
    )

    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.95))

    if cfg.output is not None:
        outfile = Path(cfg.output)
        ensure_dir(outfile.parent)
    else:
        outdir = ensure_dir(Path("plot") / cfg.system / "heatmaps" / collective.lower())
        outfile = (
            outdir
            / f"{cfg.system}_{collective.lower()}_default_vs_best_{metric}_dual_{cfg.ppn_left}ppn_{cfg.ppn_right}ppn.pdf"
        )

    written = save_figure(fig, outfile, cfg.output_format, bbox_inches="tight")
    plt.close(fig)
    return written[0]
