# Security policy

This document describes how to report security issues and what to expect from maintainers.

## Supported versions

HyperTorch is research-oriented and evolves quickly. Security fixes are provided on a best-effort basis.

| Version | Supported |
| --- | --- |
| `main` (current development) | :white_check_mark: |
| Latest released version (if any) | :white_check_mark: |
| Older releases / forks | :x: |

If you can reproduce a security issue on `main`, that is the most actionable report.

## Reporting a vulnerability

Please do **not** open a public GitHub issue for suspected vulnerabilities.

Preferred: open a **private** GitHub Security Advisory:

- https://github.com/hypernetwork-research-group/hypertorch/security/advisories/new

If you cannot use GitHub Security Advisories, contact the maintainers privately (you can find the project team contacts in [pyproject.toml](pyproject.toml)).

### What to include

To help us triage quickly, include:

- A clear description of the issue and the potential impact.
- Steps to reproduce (ideally a minimal PoC).
- Affected versions/commit SHA(s).
- Your environment (OS, Python version, install method).
- Any relevant logs, stack traces, or screenshots.

## Disclosure process

This repo follows coordinated disclosure, but timelines can vary depending on maintainer availability and severity.

## Scope

In scope:

- The `hypertorch` Python package and repository code.
- Documentation build tooling in this repo (if it can impact users).

Out of scope:

- Vulnerabilities in upstream dependencies (e.g., PyTorch, PyG, NumPy). Please report those to the upstream project.
- Vulnerabilities requiring a fully compromised environment (unless there is an additional HyperTorch-specific escalation).

## Security updates

When a fix is available, we will typically communicate via one or more of:

- A GitHub Security Advisory.
- GitHub releases / release notes (if the project is releasing versions).
