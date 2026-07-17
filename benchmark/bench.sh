#!/usr/bin/env bash
set -uo pipefail

# Default values
NUM_WORKERS=1
NUM_FEATURES=16
SEEDS=()
K_NODES=1
TEST_SET_NEGATIVE_RATIO=0.5
SPLIT_RATIOS=()
DATASETS=()
TASK=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --num-features)
            NUM_FEATURES="$2"
            shift 2
            ;;
        --seed)
            shift
            SEEDS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                SEEDS+=("$1")
                shift
            done
            ;;
        --k-nodes)
            K_NODES="$2"
            shift 2
            ;;
        --test-set-negative-ratio)
            TEST_SET_NEGATIVE_RATIO="$2"
            shift 2
            ;;
        --split-ratios)
            shift
            SPLIT_RATIOS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                SPLIT_RATIOS+=("$1")
                shift
            done
            ;;
        --datasets)
            shift
            DATASETS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                DATASETS+=("$1")
                shift
            done
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

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
        --num-workers ${NUM_WORKERS} \
        --num-features ${NUM_FEATURES} \
        --seed ${SEEDS[@]} \
        --k-nodes ${K_NODES} \
        --test-set-negative-ratio ${TEST_SET_NEGATIVE_RATIO} \
        --split-ratios ${SPLIT_RATIOS[@]} \
        --datasets ${DATASETS[@]}
else
    echo "Unknown task: $TASK" >&2
    exit 1
fi
