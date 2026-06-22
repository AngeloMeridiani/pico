# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

METRICS = ("mean", "median", "percentile_90")


def nodes_list(values: Iterable[str] | str) -> list[str]:
    if isinstance(values, str):
        return [item.strip() for item in values.split(",") if item.strip()]
    return [str(item).strip() for item in values if str(item).strip()]


def metadata_path(system: str) -> Path:
    return Path("results") / f"{system}_metadata.csv"


def read_metadata(system: str) -> pd.DataFrame:
    path = metadata_path(system)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file {path} not found.")
    return pd.read_csv(path)


def filter_metadata(
    metadata: pd.DataFrame,
    *,
    system: str,
    collective: str,
    nodes: str,
    tasks_per_node: int | None = None,
    notes: str | None = None,
    drop_ompi_bine_on_leonardo: bool = False,
) -> pd.DataFrame:
    filtered = metadata[
        (metadata["collective_type"].astype(str).str.lower() == collective.lower())
        & (metadata["nnodes"].astype(str) == str(nodes))
    ].copy()

    if tasks_per_node is not None and "tasks_per_node" in filtered.columns:
        tpn = pd.to_numeric(filtered["tasks_per_node"], errors="coerce")
        filtered = filtered[tpn == tasks_per_node]

    if notes:
        filtered = filtered[filtered["notes"].astype(str).str.strip() == notes.strip()]
    elif "notes" in filtered.columns:
        filtered = filtered[filtered["notes"].isna()]

    if drop_ompi_bine_on_leonardo and system == "leonardo" and "mpi_lib" in filtered.columns:
        filtered = filtered[~filtered["mpi_lib"].astype(str).str.contains("OMPI_BINE", case=False, na=False)]

    if filtered.empty:
        return filtered

    sort_cols = [col for col in ("timestamp", "test_id") if col in filtered.columns]
    if sort_cols:
        filtered = filtered.sort_values(sort_cols, kind="stable")
    return filtered


def discover_summary_dirs(
    *,
    system: str,
    collective: str,
    nodes: Iterable[str] | str,
    tasks_per_node: int | None = None,
    notes: str | None = None,
    drop_ompi_bine_on_leonardo: bool = False,
    require_all_nodes: bool = True,
) -> dict[str, Path]:
    metadata = read_metadata(system)
    summaries: dict[str, Path] = {}
    path = metadata_path(system)

    for node_count in nodes_list(nodes):
        filtered = filter_metadata(
            metadata,
            system=system,
            collective=collective,
            nodes=node_count,
            tasks_per_node=tasks_per_node,
            notes=notes,
            drop_ompi_bine_on_leonardo=drop_ompi_bine_on_leonardo,
        )
        if filtered.empty:
            if not require_all_nodes:
                continue
            raise RuntimeError(f"Metadata file {path} does not contain data for {collective}, nodes={node_count}.")

        latest = filtered.iloc[-1]
        summaries[node_count] = Path("results") / system / str(latest["timestamp"])

    return summaries


def ensure_summary(summary_dir: str | Path) -> Path:
    summary_dir = Path(summary_dir)
    summary_path = summary_dir / "aggregated_results_summary.csv"
    if summary_path.exists():
        return summary_path

    subprocess.run(
        [sys.executable, "./plot/summarize_data.py", "--result-dir", str(summary_dir)],
        stdout=subprocess.DEVNULL,
        check=True,
    )
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary generation did not create {summary_path}.")
    return summary_path


def load_summary_frames(
    *,
    system: str,
    collective: str,
    nodes: Iterable[str] | str,
    tasks_per_node: int | None = None,
    notes: str | None = None,
    drop_ompi_bine_on_leonardo: bool = False,
    drop_four_byte_buffers: bool = True,
    require_all_nodes: bool = True,
) -> pd.DataFrame:
    summaries = discover_summary_dirs(
        system=system,
        collective=collective,
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        notes=notes,
        drop_ompi_bine_on_leonardo=drop_ompi_bine_on_leonardo,
        require_all_nodes=require_all_nodes,
    )

    frames: list[pd.DataFrame] = []
    for node_count, summary_dir in summaries.items():
        df = pd.read_csv(ensure_summary(summary_dir))
        df = df[df["collective_type"].astype(str).str.lower() == collective.lower()].copy()
        if drop_four_byte_buffers and "buffer_size" in df.columns:
            df = df[df["buffer_size"] != 4]
        if df.empty:
            continue
        df["Nodes"] = str(node_count)
        frames.append(df)

    if not frames:
        raise RuntimeError("No data found for the requested configuration.")
    return pd.concat(frames, ignore_index=True)


def add_bandwidth_column(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    metric = metric.lower()
    if metric not in METRICS:
        raise ValueError(f"Unsupported metric '{metric}'. Expected one of {METRICS}.")
    if metric not in df.columns:
        raise KeyError(f"Metric column '{metric}' not found in dataframe.")

    out = df.copy()
    values = pd.to_numeric(out[metric], errors="coerce")
    buffers = pd.to_numeric(out["buffer_size"], errors="coerce")
    out[f"bandwidth_{metric}"] = ((buffers * 8.0) / 1e9) / (values / 1e9)
    return out
