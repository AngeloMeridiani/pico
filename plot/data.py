# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .utils import PlotMetadata


class SummaryEmptyError(RuntimeError):
    """Raised when filtering removes every row from a summary."""


def read_summary(path: str) -> pd.DataFrame:
    """
    Load an aggregated summary CSV produced by ``summarize_data.py``.
    """
    df = pd.read_csv(path)
    if df.empty:
        raise SummaryEmptyError(f"Summary file {path} is empty.")
    return df


def extract_metadata(df: pd.DataFrame) -> PlotMetadata:
    """
    Extract the invariant metadata fields from a summary dataframe.
    """
    row = df.iloc[0]
    tasks_per_node = row.get("tasks_per_node")
    return PlotMetadata(
        system=row["system"],
        timestamp=row["timestamp"],
        mpi_lib=row["mpi_lib"],
        nnodes=str(row["nnodes"]),
        tasks_per_node=int(tasks_per_node) if pd.notna(tasks_per_node) else 1,
        gpu_lib=row.get("gpu_lib", "CPU"),
    )


def filter_summary(
    df: pd.DataFrame,
    *,
    collective: str | None = None,
    datatype: str | None = None,
    algorithm: Iterable[str] | None = None,
    filter_by: Iterable[str] | None = None,
    filter_out: Iterable[str] | None = None,
    min_dim: int | None = None,
    max_dim: int | None = None,
) -> pd.DataFrame:
    """
    Apply filtering operations that mirror the legacy CLI flags.
    """
    filtered = df.copy()

    if collective:
        filtered = filtered[filtered["collective_type"] == collective]
    if datatype:
        filtered = filtered[filtered["datatype"] == datatype]
    if algorithm:
        algorithms = list(algorithm)
        filtered = filtered[filtered["algo_name"].isin(algorithms)]
    if filter_by:
        pattern = "|".join(filter_by)
        filtered = filtered[filtered["algo_name"].str.contains(pattern, case=False, na=False)]
    if filter_out:
        pattern = "|".join(filter_out)
        filtered = filtered[~filtered["algo_name"].str.contains(pattern, case=False, na=False)]
    if min_dim is not None:
        filtered = filtered[filtered["buffer_size"] >= int(min_dim)]
    if max_dim is not None:
        filtered = filtered[filtered["buffer_size"] <= int(max_dim)]

    if filtered.empty:
        raise SummaryEmptyError("Filtered data is empty.")
    return filtered


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove metadata columns that do not vary across the dataframe.
    """
    always_drop = ["array_dim"]
    conditional_drop = [
        "nnodes",
        "system",
        "timestamp",
        "test_id",
        "MPI_Op",
        "notes",
        "mpi_lib",
        "mpi_lib_version",
        "gpu_lib",
        "gpu_lib_version",
        "gpu_awareness",
        "libpico_version",
    ]

    cleaned = df.drop(columns=[col for col in always_drop if col in df.columns], errors="ignore")

    for col in conditional_drop:
        if col not in cleaned.columns:
            continue
        uniques = cleaned[col].dropna().unique()
        if len(uniques) <= 1:
            cleaned = cleaned.drop(columns=col)

    return cleaned


def normalize_dataset(
    data: pd.DataFrame,
    *,
    mpi_lib: str,
    gpu_lib: str,
    base: str | None = None,
) -> pd.DataFrame:
    """
    Normalize the dataset dividing by the reference algorithm.
    """
    df = data.copy()

    chosen_base = base
    if chosen_base is None:
        if mpi_lib in {"OMPI", "OMPI_BINE"}:
            chosen_base = "allreduce_nccl_pat" if gpu_lib == "CUDA" else "default_ompi"
        elif mpi_lib in {"MPICH", "CRAY_MPICH"}:
            chosen_base = "default_mpich"

    grouped = df.groupby("buffer_size")
    normalized_means = pd.Series(index=df.index, dtype=float)
    normalized_stds = pd.Series(index=df.index, dtype=float)

    for buffer_size, group in grouped:
        base_row = group[group["algo_name"] == chosen_base]
        if base_row.empty:
            continue
        base_mean = base_row["mean"].iloc[0]
        normalized_means.loc[group.index] = group["mean"] / base_mean
        normalized_stds.loc[group.index] = (group["std"] / group["mean"]) * normalized_means.loc[group.index]

    df["normalized_mean"] = normalized_means.fillna(1.0)
    df["normalized_std"] = normalized_stds.fillna(0.0)
    return df
