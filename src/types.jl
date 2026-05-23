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

"""
    AbstractTensor

Supertype for tensor-like objects in SymbolicTensors.jl.

Concrete subtypes are defined in later includes (e.g. [`Tensor`](@ref),
[`KroneckerDelta`](@ref)). Subtypes must implement
[`print_as`](@ref `(SymbolicTensors.print_as)(::AbstractTensor)`).
"""
abstract type AbstractTensor end