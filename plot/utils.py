from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def ensure_dir(path: str | Path) -> Path:
    """
    Create ``path`` if it does not already exist and return it as ``Path``.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def format_bytes(value: float | int | str) -> str:
    """
    Convert raw byte counts into a human friendly representation.
    """
    try:
        x = float(value)
    except (ValueError, TypeError):
        return str(value)

    if x >= 1024**2:
        return f"{x / 1024**2:.0f} MiB"
    if x >= 1024:
        return f"{x / 1024:.0f} KiB"
    return f"{x:.0f} B"


def human_readable_size(num_bytes: float | int) -> str:
    """
    Convert bytes to a <value> <unit> string using IEC units.
    """
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{int(round(value))} {unit}"
        value /= 1024.0
    return f"{int(round(value))} PiB"


def sort_key(algo: str) -> tuple[int, str]:
    """
    Stable ordering for algorithm names so plots look consistent.
    """
    if algo.startswith("default"):
        return (0, algo)
    if not algo.endswith("over"):
        return (1, algo)
    if "bine" not in algo:
        return (2, algo)
    return (3, algo)


def build_tab10_palette(sorted_algos: Iterable[str]) -> Mapping[str, tuple[float, float, float]]:
    """
    Map each algorithm to a colour from matplotlib's ``tab10`` palette.
    The palette is cycled if more than ten algorithms are requested.
    """
    colors = plt.get_cmap("tab10").colors
    return {algo: colors[i % len(colors)] for i, algo in enumerate(sorted_algos)}


def draw_errorbars(
    ax,
    data,
    sorted_algos: Iterable[str],
    std_threshold: float,
    *,
    threshold_mode: str = "absolute",
    loc: float = 0.05,
    top: bool = False,
    y_min: float = 2.0,
) -> None:
    """
    Reusable errorbar helper that supports both absolute and relative thresholds.
    """
    containers = ax.containers
    for idx, algo in enumerate(sorted_algos):
        algo_group = data[data["algo_name"] == algo]
        if idx >= len(containers):
            continue
        container = containers[idx]
        for bar, (_, row) in zip(container, algo_group.iterrows()):
            x = bar.get_x() + bar.get_width() / 2.0
            y = bar.get_height()
            std_dev = row.get("normalized_std", 0.0)

            if threshold_mode == "absolute":
                if std_dev > std_threshold:
                    ax.scatter(x, y + loc, color="red", s=50, zorder=5)
                else:
                    ax.errorbar(x, y, yerr=std_dev, fmt="none", ecolor="black", capsize=3, zorder=4)
            elif threshold_mode == "relative":
                real_threshold = std_threshold * y
                if std_dev <= real_threshold:
                    ax.errorbar(x, y, yerr=std_dev, fmt="none", ecolor="black", capsize=3, zorder=4)
                else:
                    if top and y < y_min:
                        continue
                    ax.scatter(x, y + loc, color="red", s=50, zorder=5)
            else:
                raise ValueError("threshold_mode must be 'absolute' or 'relative'")


def format_time_units_ns(value, _pos) -> str:
    """
    Format nanosecond tick labels using sensible units.
    """
    if value < 1_000:
        return f"{int(value)}ns" if float(value).is_integer() else f"{value:.1f} ns"
    if value < 1_000_000:
        val = value / 1_000
        return f"{int(val)}µs" if float(val).is_integer() else f"{val:.1f} µs"

    val = value / 1_000_000
    return f"{int(val)}ms" if float(val).is_integer() else f"{val:.1f} ms"


def build_ratio_colormap() -> LinearSegmentedColormap:
    """
    Convenience to reproduce the red -> white -> green map used by legacy heatmaps.
    """
    return LinearSegmentedColormap.from_list("RedGreen", ["darkred", "white", "darkgreen"])


@dataclass(slots=True)
class PlotMetadata:
    """
    Lightweight container for summary metadata (system, collective, ...).
    """

    system: str
    timestamp: str
    mpi_lib: str
    nnodes: int
    tasks_per_node: int
    gpu_lib: str

    @property
    def mpi_tasks(self) -> int:
        return self.nnodes * self.tasks_per_node
