# =========================================
# types.jl — SymbolicTensors.jl
#
# Primitive type aliases shared across all submodules.
# Loaded first — no dependencies.
# =========================================

"""
    Dim = Union{Int, Symbol}

The type of a manifold dimension. Either a concrete positive integer
(e.g. `4` for a 4-dimensional spacetime) or a symbolic name (e.g. `:n`)
for parametric/general-rank calculations where the dimension is not
fixed at definition time.
"""
const Dim = Union{Int, Symbol}