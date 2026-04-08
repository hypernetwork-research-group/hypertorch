import warnings

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
warnings.filterwarnings("ignore", message="`torch_geometric.distributed` has been deprecated")
warnings.filterwarnings("ignore", message=".*torch.jit.script.*is not supported in Python 3.14.*")
warnings.filterwarnings(
    "ignore", message="Sparse invariant checks are implicitly disabled:UserWarning"
)
