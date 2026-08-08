"""ECENet — an SO(3)-equivariant interatomic potential (MLIP) using per-edge
SO(2) features.

Public API:
    ECENet            — the model (message passing on when n_mp >= 2)
    ECENetCalculator  — ASE calculator wrapper (lazy import; needs `ase`)
    LESLongRange      — optional LES long-range add-on (lazy; needs `les`)
"""

from ecenet.model import ECENet

__all__ = ["ECENet", "ECENetCalculator", "LESLongRange"]


def __getattr__(name):
    # Lazy so `import ecenet` doesn't require ASE (calculator) or the optional
    # `les` package (LES add-on) unless they are used.
    if name == "ECENetCalculator":
        from ecenet.calculator import ECENetCalculator
        return ECENetCalculator
    if name == "LESLongRange":
        from ecenet.les import LESLongRange
        return LESLongRange
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
