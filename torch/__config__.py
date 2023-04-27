import torch


def show():
    """
    Return a human-readable string with descriptions of the
    configuration of PyTorch.
    """
    return torch._C._show_config()


# TODO: In principle, we could provide more structured version/config
# information here. For now only CXX_FLAGS is exposed, as Timer
# uses them.
def _cxx_flags():
    """Returns the CXX_FLAGS used when building PyTorch."""
    return torch._C._cxx_flags()


def parallel_info():
    r"""Returns detailed string with parallelization settings"""
    return torch._C._parallel_info()


def get_cpu_capability():
    r"""Returns cpu capability as a string value.

    Possible values:
    - "DEFAULT"
    - "VSX"
    - "Z VECTOR"
    - "NO AVX"
    - "AVX2"
    - "AVX512"
    """
    return torch._C._get_cpu_capability()
