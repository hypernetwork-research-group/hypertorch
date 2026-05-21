import ast
from pathlib import Path

import griffe
import pytest


HIF_MODULE = Path(__file__).resolve().parents[3] / "hyperbench/data/hif.py"


def _method_docstring(class_name: str, method_name: str) -> str:
    module = ast.parse(HIF_MODULE.read_text())
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    docstring = ast.get_docstring(item)
                    assert docstring is not None
                    return docstring

    raise AssertionError(f"{class_name}.{method_name} was not found")


@pytest.mark.parametrize(
    "method_name",
    [
        "load_from_url",
        "load_from_path",
    ],
)
def test_hif_loader_docstrings_render_google_sections(method_name: str) -> None:
    sections = griffe.Docstring(_method_docstring("HIFLoader", method_name)).parse("google")
    section_kinds = [section.kind for section in sections]

    assert griffe.DocstringSectionKind.parameters in section_kinds
    assert griffe.DocstringSectionKind.returns in section_kinds
