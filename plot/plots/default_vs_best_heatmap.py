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

from ..utils import ensure_dir, human_readable_size, save_figure
from .comparison_heatmap import BIG_FONT_SIZE, SMALL_FONT_SIZE

METRICS = ("mean", "median", "percentile_90")
DEFAULT_BASELINE_PATTERN = r"^default(?:[_-](?:ompi|mpich|nccl|default))?$"

_ALGO_CODE_HINTS: tuple[tuple[str, str], ...] = (
    ("recursive_doubling", "RD"),
    ("recursive_halving", "RH"),
    ("rabenseifner", "RB"),
    ("binomial", "BN"),
    ("binary", "BI"),
    ("knomial", "KN"),
    ("scatter_allgather", "SA"),
    ("distance_doubling", "DD"),
    ("pairwise", "PW"),
    ("neighbor", "NB"),
    ("ring", "RG"),
    ("bruck", "BR"),
    ("linear", "LN"),
    ("in_order", "IO"),
    ("sparbit", "SP"),
    ("segmented", "SG"),
    ("block_by_block", "BB"),
    ("permute_remap", "PR"),
    ("send_remap", "SR"),
    ("2_blocks", "B2"),
)

_CODE_STOPWORDS = {
    "allreduce",
    "allgather",
    "reduce",
    "scatter",
    "reduce_scatter",
    "bcast",
    "gather",
    "alltoall",
    "ompi",
    "mpich",
    "nccl",
    "mpi",
    "dtype",
    "int32",
    "float",
    "double",
    "char",
    "over",
}


@dataclass(slots=True)
class DefaultVsBestHeatmapConfig:
    system: str
    collective: str
    runs: Iterable[str]
    metric: str = "mean"
    default_pattern: str = DEFAULT_BASELINE_PATTERN
    tasks_per_node: int | None = None
    exclude: str | None = None
    title: str | None = None
    output: str | Path | None = None
    output_format: str = "pdf"


def _node_sort_key(node: str) -> tuple[int, int | str, str]:
    text = str(node).strip()
    if text.isdigit():
        return (0, int(text), text)

    parts = text.split("x")
    if parts and all(part.isdigit() for part in parts):
        prod = 1
        for part in parts:
            prod *= int(part)
        return (1, prod, text)

    return (2, text, text)


def _parse_run_entry(entry: str) -> tuple[str, str | None]:
    token = entry.strip()
    if not token:
        raise ValueError("Found an empty run token.")
    if ":" not in token:
        return token, None

    timestamp, nodes = token.split(":", 1)
    timestamp = timestamp.strip()
    nodes = nodes.strip()
    if not timestamp:
        raise ValueError(f"Invalid run token '{entry}'. Missing timestamp.")
    if not nodes:
        raise ValueError(f"Invalid run token '{entry}'. Missing node count.")
    return timestamp, nodes


def _infer_nodes(subset: pd.DataFrame) -> str:
    if "nnodes" not in subset.columns:
        raise RuntimeError("Cannot infer nodes for a run because 'nnodes' column is missing.")
    unique_nodes = (
        subset["nnodes"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .unique()
        .tolist()
    )
    if not unique_nodes:
        raise RuntimeError("Unable to infer node count from run data.")
    if len(unique_nodes) > 1:
        raise RuntimeError(f"Run mixes multiple node counts ({unique_nodes}); pass TIMESTAMP:NODES.")
    return unique_nodes[0]


def _algo_base_code(algo_name: str) -> str:
    lower = algo_name.lower()
    if "bine" in lower:
        if "permute_remap" in lower:
            return "BiP"
        if "send_remap" in lower:
            return "BiS"
        if "block_by_block" in lower:
            return "BiB"
        if "2_blocks" in lower:
            return "Bi2"
        if "segmented" in lower:
            return "BiG"
        return "Bi"

    for marker, code in _ALGO_CODE_HINTS:
        if marker in lower:
            return code

    tokens = [
        token
        for token in re.split(r"[^a-z0-9]+", lower)
        if token and token not in _CODE_STOPWORDS and not token.isdigit()
    ]
    if not tokens:
        return "??"
    if len(tokens) == 1:
        return tokens[0][:2].upper()
    return (tokens[0][0] + tokens[1][0]).upper()


def _build_algo_codes(algorithms: list[str]) -> dict[str, str]:
    used: set[str] = set()
    codes: dict[str, str] = {}
    for algo in algorithms:
        base = _algo_base_code(algo)
        code = base
        suffix = 2
        while code in used:
            code = f"{base}{suffix}"
            suffix += 1
        used.add(code)
        codes[algo] = code
    return codes


def _text_color_for_rgba(color: tuple[float, float, float, float]) -> str:
    r, g, b, _ = color
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if luminance >= 0.62 else "white"


def _format_ppn(tasks_per_node: set[float]) -> str:
    if not tasks_per_node:
        return "unknown"
    if len(tasks_per_node) == 1:
        value = next(iter(tasks_per_node))
        if float(value).is_integer():
            return str(int(value))
        return str(value)
    ordered = sorted(tasks_per_node)
    labels = [str(int(v)) if float(v).is_integer() else str(v) for v in ordered]
    return "mixed(" + ",".join(labels) + ")"


def _is_no_bine_exclude(exclude: str | None) -> bool:
    return bool(exclude and re.search(r"bine", exclude, flags=re.IGNORECASE))


def generate_default_vs_best_heatmap(cfg: DefaultVsBestHeatmapConfig) -> Path:
    metric = cfg.metric.lower()
    if metric not in METRICS:
        raise ValueError(f"Unsupported metric '{cfg.metric}'. Expected one of {METRICS}.")
    try:
        re.compile(cfg.default_pattern)
    except re.error as exc:
        raise ValueError(f"Invalid --default-pattern regex: {exc}") from exc

    collective = cfg.collective.upper()
    frames: list[pd.DataFrame] = []
    tasks_per_node_values: set[float] = set()
    selected_tasks_per_node: float | None = float(cfg.tasks_per_node) if cfg.tasks_per_node is not None else None

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

        if "tasks_per_node" in subset.columns:
            subset["tasks_per_node_value"] = pd.to_numeric(subset["tasks_per_node"], errors="coerce")
            run_ppn_values = sorted(subset["tasks_per_node_value"].dropna().unique().tolist())

            if cfg.tasks_per_node is None:
                if len(run_ppn_values) > 1:
                    raise RuntimeError(
                        f"Run {timestamp} contains multiple tasks_per_node values {run_ppn_values}. "
                        "Pass --tasks-per-node to select one and avoid mixing configurations."
                    )
                if run_ppn_values:
                    run_ppn = float(run_ppn_values[0])
                    if selected_tasks_per_node is None:
                        selected_tasks_per_node = run_ppn
                    elif not np.isclose(selected_tasks_per_node, run_ppn):
                        raise RuntimeError(
                            "Selected runs contain different tasks_per_node values "
                            f"({selected_tasks_per_node} vs {run_ppn}). "
                            "Pass --tasks-per-node to choose one."
                        )

            if selected_tasks_per_node is not None:
                subset = subset[np.isclose(subset["tasks_per_node_value"], selected_tasks_per_node)]
            if subset.empty:
                continue
            tasks_per_node_values.update(
                subset["tasks_per_node_value"].dropna().astype(float).unique().tolist()
            )
        elif cfg.tasks_per_node is not None:
            raise RuntimeError(
                f"Run {timestamp} has no tasks_per_node column, cannot apply --tasks-per-node {cfg.tasks_per_node}."
            )

        node_value = run_nodes or _infer_nodes(subset)
        subset["Nodes"] = str(node_value)
        subset["time"] = pd.to_numeric(subset[metric], errors="coerce")
        subset = subset[np.isfinite(subset["time"]) & (subset["time"] > 0)]
        if subset.empty:
            continue

        group_cols = ["buffer_size", "Nodes", "algo_name"]
        selected_cols = ["buffer_size", "Nodes", "algo_name", "time"]
        if "tasks_per_node_value" in subset.columns:
            group_cols.append("tasks_per_node_value")
            selected_cols.append("tasks_per_node_value")

        subset = subset.loc[subset.groupby(group_cols)["time"].idxmin()]
        frames.append(subset[selected_cols])

    if not frames:
        raise RuntimeError("No matching data found for the requested runs/collective.")

    data = pd.concat(frames, ignore_index=True)
    all_nodes = sorted(data["Nodes"].astype(str).unique().tolist(), key=_node_sort_key)
    all_buffers = sorted(pd.to_numeric(data["buffer_size"], errors="coerce").dropna().astype(int).unique().tolist())
    records: list[dict[str, object]] = []

    cell_group_cols = ["buffer_size", "Nodes"]
    if "tasks_per_node_value" in data.columns:
        cell_group_cols.append("tasks_per_node_value")

    for group_key, group in data.groupby(cell_group_cols):
        if len(cell_group_cols) == 3:
            buffer_size, nodes, _ = group_key
        else:
            buffer_size, nodes = group_key

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
            "No cells could be computed. Ensure each run has a default algorithm "
            f"matching pattern '{cfg.default_pattern}' and at least one non-default algorithm."
        )

    matrix = pd.DataFrame.from_records(records)
    ordered_nodes = all_nodes
    ordered_buffers = all_buffers

    ratio_data = matrix.pivot(index="buffer_size", columns="Nodes", values="ratio")
    ratio_data = ratio_data.reindex(index=ordered_buffers, columns=ordered_nodes)

    algo_data = matrix.pivot(index="buffer_size", columns="Nodes", values="best_algo")
    algo_data = algo_data.reindex(index=ordered_buffers, columns=ordered_nodes)

    unique_algos = matrix["best_algo"].drop_duplicates().tolist()
    algo_to_idx = {algo: idx for idx, algo in enumerate(unique_algos)}
    algo_codes = _build_algo_codes(unique_algos)

    numeric_data = algo_data.apply(lambda col: col.map(algo_to_idx))

    tab10 = plt.get_cmap("tab10")
    colors = [tab10(i % 10) for i in range(len(unique_algos))]
    cmap = ListedColormap(colors)
    cmap.set_bad(color="white")

    n_rows = len(ratio_data.index)
    fig_height = max(8.5, min(28.0, 0.40 * n_rows + 3.5))
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    sns.heatmap(
        numeric_data,
        cmap=cmap,
        vmin=-0.5,
        vmax=(len(unique_algos) - 1) + 0.5,
        cbar=False,
        annot=False,
        ax=ax,
    )

    for i, buffer_size in enumerate(ratio_data.index):
        for j, nodes in enumerate(ratio_data.columns):
            algo = algo_data.iloc[i, j]
            ratio = ratio_data.iloc[i, j]
            if pd.isna(algo) or pd.isna(ratio):
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    fontsize=SMALL_FONT_SIZE - 1,
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
                fontsize=BIG_FONT_SIZE - 1,
                weight="bold",
                color=text_color,
            )
            ax.text(
                j + 0.5,
                i + 0.74,
                f"{float(ratio):.2f}",
                ha="center",
                va="center",
                fontsize=SMALL_FONT_SIZE - 1,
                color=text_color,
            )

    # Keep y labels readable for large heatmaps.
    max_visible_y_labels = 12
    y_tick_stride = max(1, int(np.ceil(n_rows / max_visible_y_labels)))
    y_tick_indices = np.arange(0, n_rows, y_tick_stride, dtype=int).tolist()
    if y_tick_indices and y_tick_indices[-1] != n_rows - 1:
        y_tick_indices.append(n_rows - 1)
    y_tick_indices = np.array(sorted(set(y_tick_indices)), dtype=int)
    y_tick_labels = [human_readable_size(int(ratio_data.index[i])) for i in y_tick_indices]
    y_label_fontsize = SMALL_FONT_SIZE - 4
    y_label_fontsize = max(7, y_label_fontsize)
    ax.set_yticks(y_tick_indices + 0.5)
    ax.set_yticklabels(y_tick_labels, fontsize=y_label_fontsize, rotation=90)
    ax.set_xticks(np.arange(len(ratio_data.columns)) + 0.5)
    ax.set_xticklabels(ratio_data.columns, fontsize=SMALL_FONT_SIZE)
    ax.set_xlabel("# Nodes", fontsize=SMALL_FONT_SIZE)
    ax.set_ylabel("Vector Size", fontsize=BIG_FONT_SIZE)

    ppn_label = _format_ppn(tasks_per_node_values)
    title = cfg.title or (
        f"{cfg.system.capitalize()}, {collective.lower().capitalize()}, {ppn_label} PPN"
    )
    if _is_no_bine_exclude(cfg.exclude) and "(no Bine)" not in title:
        title = f"{title} (no Bine)"
    ax.set_title(title, fontsize=BIG_FONT_SIZE, pad=20)

    legend_handles = [
        Patch(
            facecolor=colors[algo_to_idx[algo]],
            edgecolor="black",
            label=f"{algo_codes[algo]}: {algo}",
        )
        for algo in unique_algos
    ]
    legend_ncol = min(3, max(1, len(legend_handles)))
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        borderaxespad=0.0,
        ncol=legend_ncol,
        frameon=True,
        fontsize=SMALL_FONT_SIZE - 2,
    )

    fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))

    if cfg.output is not None:
        outfile = Path(cfg.output)
        ensure_dir(outfile.parent)
    else:
        outdir = ensure_dir(Path("plot") / cfg.system / "heatmaps" / collective.lower())
        outfile = outdir / f"{cfg.system}_{collective.lower()}_default_vs_best_{metric}.pdf"

    written = save_figure(fig, outfile, cfg.output_format, bbox_inches="tight")
    plt.close(fig)
    return written[0]
