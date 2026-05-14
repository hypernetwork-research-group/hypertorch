import importlib
import warnings


warnings.filterwarnings("ignore", message=".*torch.jit.script.*is not supported in Python 3.14.*")
warnings.filterwarnings("ignore", message="`torch_geometric.distributed` has been deprecated")
warnings.filterwarnings("ignore", message=".*isinstance.*LeafSpec.*is deprecated.*")

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
warnings.filterwarnings("ignore", message="Sparse invariant checks are implicitly disabled")

__all__ = ["lightning", "torch_geometric", "torchmetrics"]


def __getattr__(name):
    if name in __all__:
        module = importlib.import_module(name)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
