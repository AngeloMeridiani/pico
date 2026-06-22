#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

SYSTEM="leonardo"
METRIC="${METRIC:-mean}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-pdf}"
PPN_LEFT="${PPN_LEFT:-1}"
PPN_RIGHT="${PPN_RIGHT:-4}"
DEFAULT_PATTERN="${DEFAULT_PATTERN:-^default(?:[_-](?:ompi|mpich|nccl|default))?$}"
EXCLUDE="${EXCLUDE:-}"
WITH_NO_BINE="${WITH_NO_BINE:-1}"

RUNS=(
  "2026_02_12___17_14_22:16"
  "2026_02_12___17_13_37:256"
  "2026_02_13___16_11_38:128"
  "2026_02_13___16_11_57:64"
  "2026_02_13___16_12_12:32"
)

DEFAULT_COLLECTIVES=(
  "ALLGATHER"
  "ALLREDUCE"
  "BCAST"
  # "GATHER"
  "REDUCE"
  "REDUCE_SCATTER"
  # "SCATTER"
)

TARGET_COLLECTIVES=("${DEFAULT_COLLECTIVES[@]}")
if [[ -n "${COLLECTIVES:-}" ]]; then
  tmp="${COLLECTIVES//,/ }"
  read -r -a TARGET_COLLECTIVES <<< "$tmp"
fi

echo "Using system=${SYSTEM}, metric=${METRIC}"
echo "Output format: ${OUTPUT_FORMAT}"
echo "PPN pair: left=${PPN_LEFT}, right=${PPN_RIGHT}"
echo "Runs: ${RUNS[*]}"
echo "Collectives: ${TARGET_COLLECTIVES[*]}"
echo "Default pattern: ${DEFAULT_PATTERN}"
echo "Extra exclude: ${EXCLUDE:-<none>}"
echo "Generate no-bine variant: ${WITH_NO_BINE}"

for run in "${RUNS[@]}"; do
  ts="${run%%:*}"
  result_dir="results/${SYSTEM}/${ts}"
  summary_file="${result_dir}/aggregated_results_summary.csv"
  if [[ ! -d "${result_dir}" ]]; then
    echo "Missing results directory: ${result_dir}" >&2
    exit 1
  fi
  if [[ ! -f "${summary_file}" ]]; then
    echo "Summarizing ${result_dir}..."
    python3 ./plot/summarize_data.py --result-dir "${result_dir}"
  fi
done

for collective in "${TARGET_COLLECTIVES[@]}"; do
  collective_lower="${collective,,}"
  outdir="plot/${SYSTEM}/heatmaps/${collective_lower}"
  mkdir -p "${outdir}"

  outfile_base="${outdir}/${SYSTEM}_${collective_lower}_default_vs_best_${METRIC}_dual_${PPN_LEFT}ppn_${PPN_RIGHT}ppn.pdf"
  common_cmd=(
    python3 -m plot default-vs-best-heatmap-dual
    --system "${SYSTEM}"
    --collective "${collective}"
    --runs "${RUNS[@]}"
    --ppn-left "${PPN_LEFT}"
    --ppn-right "${PPN_RIGHT}"
    --metric "${METRIC}"
    --default-pattern "${DEFAULT_PATTERN}"
    --output-format "${OUTPUT_FORMAT}"
  )

  base_cmd=("${common_cmd[@]}" --output "${outfile_base}")
  if [[ -n "${EXCLUDE}" ]]; then
    base_cmd+=(--exclude "${EXCLUDE}")
  fi
  echo "Generating ${collective} -> ${outfile_base}"
  "${base_cmd[@]}"

  if [[ "${WITH_NO_BINE}" == "1" ]]; then
    outfile_no_bine="${outdir}/${SYSTEM}_${collective_lower}_default_vs_best_${METRIC}_dual_${PPN_LEFT}ppn_${PPN_RIGHT}ppn_no_bine.pdf"
    no_bine_exclude="bine"
    if [[ -n "${EXCLUDE}" ]]; then
      no_bine_exclude="${EXCLUDE}|bine"
    fi
    no_bine_cmd=("${common_cmd[@]}" --exclude "${no_bine_exclude}" --output "${outfile_no_bine}")
    echo "Generating ${collective} (no-bine) -> ${outfile_no_bine}"
    "${no_bine_cmd[@]}"
  fi
done

echo "Done."
