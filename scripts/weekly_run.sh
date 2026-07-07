#!/usr/bin/env bash

set -uo pipefail

# TAG_TO_CHECK=$(git tag --sort=-creatordate | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -n 1) #TODO: when first stable release, change to latest tag not beta
TAG_TO_CHECK="$1" #latest tag including beta
COMMIT_TO_CHECK="$2"

# remove first character 'v' from TAG_TO_CHECK
TAG_WITHOUT_V="${TAG_TO_CHECK:1}"

curl -L \
  "https://github.com/hypernetwork-research-group/hypertorch/archive/refs/tags/${TAG_TO_CHECK}.tar.gz" \
  -o hypertorch.tar.gz

tar -xzf hypertorch.tar.gz --strip-components=1 \
  "hypertorch-${TAG_WITHOUT_V}/examples"

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
            failed_examples+=("${example}")
        fi
    else # Windows
        if uv run python "${example}"; then
            echo "=== Passed ${example} ==="
        else
            status=$?
            echo "=== Failed ${example} with exit code ${status} ===" >&2
            failed_examples+=("${example}")
        fi
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
