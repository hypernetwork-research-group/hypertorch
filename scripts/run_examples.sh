#!/usr/bin/env bash

set -uo pipefail

if (($# > 1)); then
    echo "Usage: $0 [examples-folder]" >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

cd "${repo_root}" || exit 1

examples_folder="${1:-examples}"

if [[ ! -d "${examples_folder}" ]]; then
    echo "Examples folder not found: ${examples_folder}" >&2
    exit 1
fi

examples=()
while IFS= read -r -d '' example; do
    examples+=("${example}")
done < <(find "${examples_folder}" -type f -name "*.py" -print | LC_ALL=C sort | tr '\n' '\0')

if [[ ${#examples[@]} -eq 0 ]]; then
    echo "No Python examples found under ${examples_folder}/." >&2
    exit 1
fi

failed_examples=()

for example in "${examples[@]}"; do
    echo "=== ${example} ==="
    if make run "${example}"; then
        echo "=== Passed ${example} ==="
    else
        status=$?
        echo "=== Failed ${example} with exit code ${status} ===" >&2
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
