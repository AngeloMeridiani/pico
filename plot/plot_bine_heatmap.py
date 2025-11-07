#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

# These imports are used in your existing repo.
from plot.plots.comparison_heatmap import ComparisonHeatmapConfig, BIG_FONT_SIZE, SMALL_FONT_SIZE
from plot.utils import ensure_dir, human_readable_size

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render 'best Bine variant' heatmap for a collective.")
    p.add_argument("--system", default="leonardo", help="System name used in results/ path and plot output.")
    p.add_argument("--collective", choices=["ALLGATHER", "REDUCE_SCATTER"], required=True)
    p.add_argument("--metric", choices=["mean", "median"], default="mean")
    p.add_argument("--runs", nargs="+", metavar="TIMESTAMP:NODES",
                   help="List like 2025_04_06___13_24_31:64 (timestamp:nodes). If omitted, a Leonardo default set is used.")
    p.add_argument("--output", default=None, help="Optional explicit output PDF path. Defaults to plot/<system>/heatmaps/<collective>/best_bine_variant.pdf")
    return p

def default_runs() -> list[tuple[str, int]]:
    # Matches the RUNS used in your current scripts
    return [
        ("2025_04_05___23_20_55", 256),
        ("2025_04_06___13_24_12", 128),
        ("2025_04_06___13_24_31", 64),
        ("2025_04_06___13_24_51", 32),
        ("2025_04_06___13_25_12", 16),
        ("2025_04_06___13_25_25", 8),
        ("2025_04_06___13_25_39", 4),
    ]

def parse_runs(arg_runs: list[str] | None) -> list[tuple[str, int]]:
    if not arg_runs:
        return default_runs()
    out: list[tuple[str, int]] = []
    for item in arg_runs:
        ts, nodes_s = item.split(":", 1)
        out.append((ts, int(nodes_s)))
    return out

BASELINE_DEFAULTS = {
    "ALLGATHER": "recursive_doubling_ompi",
    "REDUCE_SCATTER": "recursive_halving_ompi",
}

BASELINE_OVERRIDES = {
    "lumi": {
        "ALLGATHER": "recursive_doubling_mpich",
        "REDUCE_SCATTER": "recursive_halving_mpich",
    }
}

def resolve_baseline(system: str, collective: str) -> str:
    system_key = system.lower()
    collective_key = collective.upper()
    override = BASELINE_OVERRIDES.get(system_key, {}).get(collective_key)
    if override:
        return override
    return BASELINE_DEFAULTS[collective_key]

def main() -> None:
    args = build_parser().parse_args()
    runs = parse_runs(args.runs)

    baseline = resolve_baseline(args.system, args.collective)

    # Category patterns (include 'bine_2_blocks' only for ALLGATHER)
    bine_patterns = ["bine_block_by_block", "bine_permute_remap", "bine_send_remap"]
    if args.collective == "ALLGATHER":
        bine_patterns.append("bine_2_blocks")

    cfg = ComparisonHeatmapConfig(
        system=args.system,
        collective=args.collective,
        nnodes=[str(n) for _, n in runs],
        target_algo=baseline,
        metric=args.metric,
        show_names=False,
        output_dir=None,
    )

    frames: list[pd.DataFrame] = []
    patterns = tuple(bine_patterns + [baseline])

    for timestamp, nodes in runs:
        summary_path = Path("results") / args.system / timestamp / "aggregated_results_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Summary file {summary_path} not found.")
        df = pd.read_csv(summary_path)
        subset = df[df["collective_type"].str.lower() == args.collective.lower()].copy()
        subset["Nodes"] = str(nodes)
        mask = subset["algo_name"].str.contains("|".join(patterns), case=False, na=False)
        subset = subset[mask]
        # Filter out dtype variants if present (matches your allgather script)
        subset = subset[~subset["algo_name"].str.contains("dtype", case=False, na=False)]
        if subset.empty:
            continue
        subset[f"bandwidth_{args.metric}"] = ((subset["buffer_size"] * 8.0) / 1e9) / (subset[args.metric].astype(float) / 1e9)
        frames.append(subset[["buffer_size", "Nodes", "algo_name", f"bandwidth_{args.metric}"]])

    if not frames:
        raise RuntimeError("No matching data found for the requested runs/collective.")

    bandwidth_df = pd.concat(frames, ignore_index=True)

    # Preserve your node ordering helper from ComparisonHeatmapConfig
    unique_nodes = [str(n) for n in bandwidth_df["Nodes"].unique()]

    def sort_nodes_desc(nodes: list[str]) -> list[str]:
        return sorted(nodes, key=lambda n: int(n))

    ordered_nodes = [node for node in cfg.sorted_nodes() if node in unique_nodes]
    if ordered_nodes:
        ordered_nodes = sort_nodes_desc(ordered_nodes)
    else:
        ordered_nodes = sort_nodes_desc(unique_nodes)

    def classify_algo(name: str) -> str | None:
        lower = name.lower()
        if "bine_permute_remap" in lower:
            return "permute"
        if "bine_send_remap" in lower:
            return "send"
        if "bine_block_by_block" in lower:
            return "block"
        if args.collective == "ALLGATHER" and "bine_2_blocks" in lower:
            return "two"
        return None

    # Build best-per-cell category and ratio maps (best Bine vs baseline)
    records: list[dict] = []
    ratio_map: dict[tuple[int, str], float] = {}

    for (buffer_size, node), group in bandwidth_df.groupby(["buffer_size", "Nodes"]):
        bine_group = group[group["algo_name"].str.contains("bine", case=False, na=False)]
        target_rows = group[group["algo_name"].str.contains(baseline, case=False, na=False)]
        if bine_group.empty or target_rows.empty:
            continue

        best_row = bine_group.loc[bine_group[f"bandwidth_{args.metric}"].idxmax()]
        category = classify_algo(best_row["algo_name"])
        if category is None:
            continue
        target_value = target_rows[f"bandwidth_{args.metric}"].max()
        if not np.isfinite(target_value) or target_value <= 0:
            continue
        ratio_map[(buffer_size, node)] = best_row[f"bandwidth_{args.metric}"] / target_value
        records.append({"buffer_size": buffer_size, "Nodes": node, "category": category})

    if not records:
        raise RuntimeError("Unable to determine a winning Bine algorithm for any cell.")

    # Encode categories to integers for heatmap colors
    letter_map = {"permute": "P", "send": "S", "block": "B"}
    code_map = {"permute": 0, "send": 1, "block": 2}
    if args.collective == "ALLGATHER":
        letter_map["two"] = "T"
        code_map["two"] = 3

    matrix_df = pd.DataFrame(records)
    matrix_df["code"] = matrix_df["category"].map(code_map)
    category_map = {(row["buffer_size"], row["Nodes"]): row["category"] for _, row in matrix_df.iterrows()}

    heatmap_data = matrix_df.pivot(index="buffer_size", columns="Nodes", values="code")
    heatmap_data = heatmap_data.reindex(index=sorted(heatmap_data.index), columns=ordered_nodes)

    # Colormap (tab10 first N colors)
    tab10 = plt.get_cmap("tab10")
    colors = [tab10(i) for i in range(len(code_map))]
    cmap = ListedColormap(colors)
    cmap.set_bad(color='white')

    plt.figure(figsize=(9.0, 6.0))
    ax = sns.heatmap(
        heatmap_data,
        cmap=cmap,
        vmin=-0.5,
        vmax=(len(code_map) - 1) + 0.5,
        cbar=False,
        annot=False,
    )

    # Draw letter + percent improvement vs baseline in each cell
    for i, buffer_size in enumerate(heatmap_data.index):
        for j, node in enumerate(heatmap_data.columns):
            code = heatmap_data.iloc[i, j]
            if np.isnan(code):
                ax.text(j + 0.5, i + 0.5, "N/A", ha="center", va="center", color="black", fontsize=SMALL_FONT_SIZE)
                continue
            category = category_map.get((buffer_size, node))
            if category is None:
                ax.text(j + 0.5, i + 0.5, "?", ha="center", va="center", color="white", fontsize=BIG_FONT_SIZE, weight="bold")
                continue
            letter = letter_map[category]
            ratio = ratio_map.get((buffer_size, node))
            ax.text(j + 0.5, i + 0.38, letter, ha="center", va="center", color="white", fontsize=BIG_FONT_SIZE, weight="bold")
            if ratio is not None and np.isfinite(ratio):
                ax.text(j + 0.5, i + 0.74, f"{ratio * 100:.0f}%", ha="center", va="center", color="white", fontsize=SMALL_FONT_SIZE - 1)

    buffer_labels = [human_readable_size(int(size)) for size in heatmap_data.index]
    ax.set_yticks(np.arange(len(buffer_labels)) + 0.5)
    ax.set_yticklabels(buffer_labels, fontsize=SMALL_FONT_SIZE)
    ax.set_xticks(np.arange(len(heatmap_data.columns)) + 0.5)
    ax.set_xticklabels(heatmap_data.columns, fontsize=SMALL_FONT_SIZE)

    ax.set_xlabel("# Nodes", fontsize=BIG_FONT_SIZE)
    ax.set_ylabel("Vector Size", fontsize=BIG_FONT_SIZE)

    plt.tight_layout()

    if args.output:
        outfile = Path(args.output)
        ensure_dir(outfile.parent)
    else:
        outdir = ensure_dir(Path("plot") / args.system / "heatmaps" / args.collective.lower())
        outname = args.system + "_" + args.collective.lower() +"_best_bine_variant.pdf"
        outfile = outdir / outname

    plt.savefig(outfile, bbox_inches="tight")
    plt.close()
    print(f"Heatmap written to {outfile}")

if __name__ == "__main__":
    main()
