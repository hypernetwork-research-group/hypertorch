from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable, Sequence

import griffe


DOCSTRING_STYLE = "google"


@dataclass(frozen=True)
class DocstringIssue:
    level: str
    message: str


class _GriffeLogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.issues: list[DocstringIssue] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.issues.append(
            DocstringIssue(
                level=record.levelname.lower(),
                message=record.getMessage(),
            )
        )


def iter_docstrings(obj: griffe.Object) -> Iterable[griffe.Docstring]:
    seen: set[int] = set()

    def walk(node: griffe.Object) -> Iterable[griffe.Docstring]:
        node_id = id(node)
        if node_id in seen:
            return
        seen.add(node_id)

        if node.docstring is not None:
            yield node.docstring

        for member in node.members.values():
            if isinstance(member, griffe.Alias):
                continue
            yield from walk(member)

    yield from walk(obj)


def _ensure_object(root: griffe.Object | griffe.Alias) -> griffe.Object:
    if isinstance(root, griffe.Alias):
        return root.final_target
    return root


def validate_docstrings(
    package: str = "hyperbench",
    *,
    repo_root: Path | None = None,
    search_paths: Sequence[Path] | None = None,
) -> list[DocstringIssue]:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]

    if search_paths is None:
        search_paths = (repo_root,)

    handler = _GriffeLogCapture()
    logger = logging.getLogger("griffe")
    previous_level = logger.level

    logger.setLevel(logging.WARNING)
    logger.addHandler(handler)

    try:
        module = griffe.load(
            package,
            search_paths=search_paths,
            docstring_parser=DOCSTRING_STYLE,
            docstring_options={
                "warn_unknown_params": True,
                "warn_missing_types": True,
                "warnings": True,
            },
        )
        root = _ensure_object(module)
        for docstring in iter_docstrings(root):
            docstring.parse()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)

    return handler.issues


def format_issues(issues: Sequence[DocstringIssue]) -> str:
    if not issues:
        return "No docstring issues found."

    lines = ["Docstring issues:"]
    for issue in issues:
        lines.extend([f"  - [{issue.level}] {issue.message}"])
    return "\n".join(lines)


def main() -> int:
    issues = validate_docstrings()
    print(format_issues(issues))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
