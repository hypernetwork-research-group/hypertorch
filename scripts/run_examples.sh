#!/usr/bin/env bash

set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

cd "${repo_root}" || exit 1

examples=(examples/**/*.py)

if [[ ${#examples[@]} -eq 0 || ! -e "${examples[0]}" ]]; then
    echo "No Python examples found under examples/." >&2
    exit 1
fi

failed_examples=()

for example in "${examples[@]}"; do
    echo "=== ${example} ==="
    if [[ "$OSTYPE" == "darwin"* || "$OSTYPE" == "linux-gnu"* ]]; then
        if python3 "${example}"; then
            echo "=== Passed ${example} ==="
        else
            status=$?
            echo "=== Failed ${example} with exit code ${status} ===" >&2
        fi
    else # Windows
        if uv run python "${example}"; then
            echo "=== Passed ${example} ==="
        else
            status=$?
            echo "=== Failed ${example} with exit code ${status} ===" >&2
        fi
    fi
        failed_examples+=("${example}")
    fi
done

if ((${#failed_examples[@]} > 0)); then
    echo "The following examples failed:" >&2
    for example in "${failed_examples[@]}"; do
        echo "  - ${example}" >&2
    done
    exit 1
fi

echo "All examples completed successfully."
