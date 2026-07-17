#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

cd "${repo_root}"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <task> -- [arguments...]" >&2
    exit 1
fi

task="$1"
shift

# Remove the optional argument separator.
if [[ "${1:-}" == "--" ]]; then
    shift
fi

case "${task}" in
    hlp)
        echo "Running HLP benchmark..."
        exec uv run python3 "${repo_root}/benchmark/bench_hlp.py" "$@"
        ;;
    *)
        echo "Unknown task: ${task}" >&2
        exit 1
        ;;
esac
