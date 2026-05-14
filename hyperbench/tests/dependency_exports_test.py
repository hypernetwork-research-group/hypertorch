import sys
import types

import pytest

import hyperbench


@pytest.mark.parametrize("name", ["lightning", "torch_geometric", "torchmetrics"])
def test_dependency_modules_are_exported(monkeypatch, name):
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    hyperbench.__dict__.pop(name, None)

    assert getattr(hyperbench, name) is module
