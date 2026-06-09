#!/usr/bin/env bash

find . -type f -name "*.py" \
    -not -path "*/.venv/*" \
    -not -path "*/__init__.py" \
    -not -path "*/examples/*" \
    -not -path "*/integration_tests/*" \
    -not -path "*/scripts/*" \
    -not -path "*/tests/*" \
    -print0 \
    | xargs -0 awk '\
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
