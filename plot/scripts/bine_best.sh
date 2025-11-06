#!/usr/bin/env bash
set -euo pipefail

METRIC="${METRIC:-mean}"
SYSTEM="${SYSTEM:-leonardo}"
RUNS_STR="${RUNS:-}"


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"


# Convert RUNS string into --runs args if provided
RUN_ARGS=()
if [[ -n "${RUNS_STR}" ]]; then
# Expecting: "2025_04_06___13_24_31:64 2025_04_06___13_24_51:32 ..."
for part in ${RUNS_STR}; do
RUN_ARGS+=(--runs "$part")
done
fi


# Pre-summarize data, skipping if summary file already exists
if [[ -n "${RUNS_STR}" ]]; then
# Only summarize those explicitly requested
for pair in ${RUNS_STR}; do
ts="${pair%%:*}"
result_dir="results/${SYSTEM}/${ts}"
summary_file="$result_dir/aggregated_results_summary.csv"

if [[ -d "$result_dir" ]]; then
if [[ ! -f "$summary_file" ]]; then
echo "Summarizing data for ${ts}..."
python3 ./plot/summarize_data.py --result-dir "$result_dir"
else
echo "Skipping summarization for ${ts} (already exists)."
fi
else
echo "Missing results directory for ${ts}" >&2
exit 1
fi
done
else
# Summarize the default runs used by the Python script
for ts in 2025_04_05___23_20_55 2025_04_06___13_24_12 2025_04_06___13_24_31 2025_04_06___13_24_51 2025_04_06___13_25_12 2025_04_06___13_25_25 2025_04_06___13_25_39; do
result_dir="results/${SYSTEM}/${ts}"
summary_file="$result_dir/aggregated_results_summary.csv"

if [[ -d "$result_dir" ]]; then
if [[ ! -f "$summary_file" ]]; then
echo "Summarizing data for ${ts}..."
python3 ./plot/summarize_data.py --result-dir "$result_dir"
else
echo "Skipping summarization for ${ts} (already exists)."
fi
else
echo "Missing results directory for ${ts}" >&2
exit 1
fi
done
fi


# Call the single Python script with the chosen options
python3 ./plot/plot_bine_heatmap.py \
--system "${SYSTEM}" \
--collective ALLGATHER \
--metric "${METRIC}" \
"${RUN_ARGS[@]}"


python3 ./plot/plot_bine_heatmap.py \
--system "${SYSTEM}" \
--collective REDUCE_SCATTER \
--metric "${METRIC}" \
"${RUN_ARGS[@]}"
