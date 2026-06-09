#!/usr/bin/env bash

find_py_files() {
	find . -type f -name "*.py" \
		-not -path "*/.venv/*" \
		-not -path "*/__init__.py" \
		-not -path "*/examples/*" \
		-not -path "*/integration_tests/*" \
		-not -path "*/scripts/*" \
		-not -path "*/tests/*" \
        "$@"
}

echo "Counting lines of code in the following Python files:"
find_py_files

LOC_COUNT=$(
    find_py_files -print0 | xargs -0 awk '\
        FNR==1 { in_docstring = 0 } \
        /^[[:blank:]]*#/ { next } \
        /"""/ { \
            n = gsub(/"""/, "&"); \
            if (n % 2 == 1) in_docstring = !in_docstring; \
            next; \
        } \
        !in_docstring { count++ } \
        END { print count } \
    '
)
echo "=$LOC_COUNT lines of code (excluding comments and docstrings)"
