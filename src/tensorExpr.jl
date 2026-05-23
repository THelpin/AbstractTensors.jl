# =========================================
# tensorExpr.jl — SymbolicTensors.jl
#
# Tensor algebra AST: TensorTerm (coeff × body) and TensorSum (finite sums).
#
# Algebraic model: free module ⊕ (ScalarLike ⊗ TensorComponent).
# Merge uses syntactic [`TensorComponent`](@ref) equality (`tensor ===`, `indices ==`).
#
# Depends on: scalar.jl, tensorComponents.jl
# Future: contractions.jl, TensorProduct
# =========================================


# =========================================
# 1.  AbstractTensorExpr and TensorTerm
# =========================================

"""
    AbstractTensorExpr

Supertype for tensor algebra expressions built from [`TensorTerm`](@ref) and
[`TensorSum`](@ref). Distinct from [`AbstractTensor`](@ref) (schema layer).
"""
abstract type AbstractTensorExpr end

"""
    TensorTerm{C, B}

Elementary tensor expression `coeff * body` in the free module over
[`TensorComponent`](@ref) basis elements.

- `coeff` : scalar coefficient ([`ScalarLike`](@ref) or Symbolics types via extension)
- `body`  : indexed tensor occurrence (typically [`TensorComponent`](@ref))
"""
struct TensorTerm{C, B} <: AbstractTensorExpr
    coeff::C
    body::B
end

"""
    term(body) -> TensorTerm

Wrap a [`TensorComponent`](@ref) with unit coefficient.
"""
term(body::TensorComponent) = TensorTerm(one_scalar(1), body)

coeff_of(t::TensorTerm) = t.coeff
body_of(t::TensorTerm)  = t.body


# =========================================
# 2.  TensorSum — sole merge bottleneck
# =========================================

"""
    TensorSum{T<:TensorTerm}

Finite sum of [`TensorTerm`](@ref)s. Normalization (merge by body, drop zero
coeffs) runs only in the inner constructor.

The zero element of the module is `TensorSum([])` — use [`is_zero`](@ref), not
`== 0`. Algebra methods never return bare scalar `0`.
"""
struct TensorSum{T <: TensorTerm} <: AbstractTensorExpr
    terms::Vector{T}

    function TensorSum(raw_terms::AbstractVector)
        if isempty(raw_terms)
            return new{TensorTerm{Int, TensorComponent}}(
                TensorTerm{Int, TensorComponent}[]
            )
        end
        terms = Vector{TensorTerm}(raw_terms)
        final_terms = _merge_terms(terms)
        if isempty(final_terms)
            return new{TensorTerm{Int, TensorComponent}}(
                TensorTerm{Int, TensorComponent}[]
            )
        end
        TT = eltype(final_terms)
        return new{TT}(final_terms)
    end
end

"""
    is_zero(s::TensorSum) -> Bool

`true` when `s` is the zero element of the tensor algebra (`TensorSum([])`).
Not the same as scalar `0`.
"""
is_zero(s::TensorSum) = isempty(s.terms)

terms_of(s::TensorSum) = s.terms


# =========================================
# 3.  Internal: collect and merge
# =========================================

_collect_terms(t::TensorTerm) = [t]
_collect_terms(s::TensorSum) = collect(s.terms)

"""
    _merge_terms(raw_terms) -> Vector{TensorTerm}

Merge by `body` using `Dict` keyed by body (`==` authoritative on collision).
Drop zero coeffs.
"""
function _merge_terms(raw_terms::Vector{<:TensorTerm})
    isempty(raw_terms) && return TensorTerm[]
    B = typeof(raw_terms[1].body)
    C = typeof(raw_terms[1].coeff)
    for t in raw_terms[2:end]
        C = promote_type(C, typeof(t.coeff))
        B = typejoin(B, typeof(t.body))
    end
    merged = Dict{B, C}()
    for t in raw_terms
        b = t.body
        c = t.coeff
        if haskey(merged, b)
            merged[b] = scalar_add(merged[b], c)
        else
            merged[b] = c
        end
    end
    out = TensorTerm{C, B}[]
    for (b, c) in merged
        is_scalar_zero(c) && continue
        push!(out, TensorTerm(c, b))
    end
    return out
end


# =========================================
# 4.  Addition
# =========================================

Base.:+(a::TensorTerm, b::TensorTerm) = TensorSum(TensorTerm[a, b])
Base.:+(a::TensorTerm, b::TensorSum) =
    TensorSum(vcat(_collect_terms(a), _collect_terms(b)))
Base.:+(a::TensorSum, b::TensorTerm) =
    TensorSum(vcat(_collect_terms(a), _collect_terms(b)))
Base.:+(a::TensorSum, b::TensorSum) =
    TensorSum(vcat(_collect_terms(a), _collect_terms(b)))

function Base.:+(::ScalarLike, ::AbstractTensorExpr)
    throw(ArgumentError(
        "Cannot add scalar coefficient to tensor expression without a body. " *
        "Use TensorTerm(coeff, body) or coeff * body."
    ))
end

function Base.:+(::AbstractTensorExpr, ::ScalarLike)
    throw(ArgumentError(
        "Cannot add tensor expression to scalar coefficient without a body. " *
        "Use TensorTerm(coeff, body) or coeff * body."
    ))
end


# =========================================
# 5.  Scalar multiplication
# =========================================

Base.:*(c::ScalarLike, comp::TensorComponent) = TensorTerm(c, comp)
Base.:*(comp::TensorComponent, c::ScalarLike) = TensorTerm(c, comp)

Base.:*(c::ScalarLike, t::TensorTerm) =
    TensorTerm(scalar_mul(c, t.coeff), t.body)
Base.:*(t::TensorTerm, c::ScalarLike) =
    TensorTerm(scalar_mul(t.coeff, c), t.body)

Base.:*(c::ScalarLike, s::TensorSum) =
    TensorSum([TensorTerm(scalar_mul(c, t.coeff), t.body) for t in s.terms])
Base.:*(s::TensorSum, c::ScalarLike) =
    TensorSum([TensorTerm(scalar_mul(c, t.coeff), t.body) for t in s.terms])

function Base.:*(::TensorTerm, ::TensorTerm)
    throw(ArgumentError(
        "Multiplication of multiple tensor components requires a `TensorProduct` " *
        "layer, which is out of scope for this version."
    ))
end


# =========================================
# 6.  Unary minus
# =========================================

Base.:-(t::TensorTerm) =
    TensorTerm(scalar_mul(-1, t.coeff), t.body)
Base.:-(s::TensorSum) =
    TensorSum([TensorTerm(scalar_mul(-1, t.coeff), t.body) for t in s.terms])


# =========================================
# 7.  Display (MIME only)
# =========================================

function _format_term_plain(t::TensorTerm)
    c = t.coeff
    if c == 1 || c == 1.0
        return string(t.body)
    end
    return "$(c) * $(t.body)"
end

function _format_term_latex(t::TensorTerm)
    c = t.coeff
    body = t.body
    if c == 1 || c == 1.0
        return _format_latex(body)
    end
    if body isa TensorComponent
        return "$(c)\\,\\($(_format_latex(body))\\)"
    end
    return "$(c) \\cdot $(body)"
end

function Base.show(io::IO, ::MIME"text/plain", t::TensorTerm)
    print(io, _format_term_plain(t))
end

function Base.show(io::IO, ::MIME"text/plain", s::TensorSum)
    if is_zero(s)
        print(io, "TensorSum([])")
        return
    end
    parts = map(_format_term_plain, s.terms)
    print(io, join(parts, " + "))
end

function Base.show(io::IO, ::MIME"text/latex", t::TensorTerm)
    print(io, "\$", _format_term_latex(t), "\$")
end

function Base.show(io::IO, ::MIME"text/latex", s::TensorSum)
    if is_zero(s)
        print(io, "\$0_{\\text{tensor}}\$")
        return
    end
    parts = map(_format_term_latex, s.terms)
    print(io, "\$", join(parts, " + "), "\$")
end

function Base.show(io::IO, ::MIME"text/html", t::TensorTerm)
    c = t.coeff
    body = t.body
    if c == 1 || c == 1.0
        print(io, _format_html(body))
        return
    end
    if body isa TensorComponent
        print(io, "<span>", c, " · </span>", _format_html(body))
    else
        print(io, "<span>", c, " · ", body, "</span>")
    end
end

function Base.show(io::IO, ::MIME"text/html", s::TensorSum)
    if is_zero(s)
        print(io, "<span><i>0</i> (empty tensor sum)</span>")
        return
    end
    first = true
    for t in s.terms
        if !first
            print(io, " + ")
        end
        show(io, MIME"text/html"(), t)
        first = false
    end
end


# =========================================
# Exports
# =========================================

export AbstractTensorExpr, TensorTerm, TensorSum
export term, coeff_of, body_of, terms_of, is_zero
