# =========================================
# scalar.jl — SymbolicTensors.jl
#
# Scalar coefficient layer for tensor algebra (TensorComponentTerm coeffs).
# No Symbolics import here — extension adds Symbolics methods.
# =========================================

"""
    ScalarLike

Coefficient types supported in core: numeric reals/complexes and symbolic
dimensions (`Symbol`, e.g. parametric bundle rank `:n`).
"""
const ScalarLike = Union{Real, Complex{<:Real}, Symbol}

"""
    is_scalar_like(x) -> Bool

Return `true` if `x` is a core [`ScalarLike`](@ref) coefficient.
The Symbolics package extension adds further methods.
"""
is_scalar_like(x) = x isa ScalarLike

# Homogeneous base cases
scalar_add(a::T, b::T) where {T} = a + b
scalar_mul(a::T, b::T) where {T} = a * b

# Heterogeneous fallback (e.g. Int + Float64)
scalar_add(a, b) = scalar_add(promote(a, b)...)
scalar_mul(a, b) = scalar_mul(promote(a, b)...)

is_scalar_zero(a::T) where {T} = a == zero(a)
is_scalar_zero(::Symbol) = false

one_scalar(::Symbol) = 1
one_scalar(x) = one(x)

export ScalarLike
export scalar_add, scalar_mul, is_scalar_zero, one_scalar, is_scalar_like
