#!/usr/bin/env bash

set -uo pipefail

TASK=$1

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

cd "${repo_root}" || exit 1

if [[ -z "$TASK" ]]; then
    echo "Usage: $0 <task>" >&2
    exit 1
fi

if [[ "$TASK" == "hlp" ]]; then
    echo "Running HLP benchmark..."
    uv run python3 ${repo_root}/benchmark/bench_hlp.py \
        --num-workers 8 \
        --num-features 32 \
        --seed 42 43 44 \
        --k-nodes 2 \
        --test-set-negative-ratio 0.6 \
        --split-ratios 0.7 0.1 0.2 \
        --datasets cora
        # --datasets cora citeseer pubmed
else
    echo "Unknown task: $TASK" >&2
    exit 1
fi
