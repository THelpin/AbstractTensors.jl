# =========================================
# tensorComponentExpr.jl — SymbolicTensors.jl
#
# Tensor algebra AST: TensorComponentTerm (coeff × body) and TensorComponentSum.
#
# Algebraic model: free module ⊕ (ScalarLike ⊗ TensorComponent).
# Merge uses syntactic [`TensorComponent`](@ref) equality (`tensor ===`, `indices ==`).
#
# Depends on: scalar.jl, tensorComponents.jl
# Future: contractions.jl, TensorProduct
# =========================================


# =========================================
# 1.  AbstractTensorComponentExpr and TensorComponentTerm
# =========================================

"""
    AbstractTensorComponentExpr

Supertype for tensor algebra expressions built from [`TensorComponentTerm`](@ref)
and [`TensorComponentSum`](@ref). Distinct from [`AbstractTensor`](@ref) (schema layer).
"""
abstract type AbstractTensorComponentExpr end

"""
    TensorComponentTerm{C, B}

Elementary tensor expression `coeff * body` in the free module over
[`TensorComponent`](@ref) basis elements.

- `coeff` : scalar coefficient ([`ScalarLike`](@ref) or Symbolics types via extension)
- `body`  : indexed tensor occurrence (typically [`TensorComponent`](@ref))
"""
struct TensorComponentTerm{C, B} <: AbstractTensorComponentExpr
    coeff::C
    body::B
end

"""
    term(body) -> TensorComponentTerm

Wrap a [`TensorComponent`](@ref) with unit coefficient.
"""
term(body::TensorComponent) = TensorComponentTerm(one_scalar(1), body)

coeff_of(t::TensorComponentTerm) = t.coeff
body_of(t::TensorComponentTerm)  = t.body


# =========================================
# 2.  TensorComponentSum — sole merge bottleneck
# =========================================

"""
    TensorComponentSum{T<:TensorComponentTerm}

Finite sum of [`TensorComponentTerm`](@ref)s. Normalization (merge by body,
drop zero coeffs) runs only in the inner constructor.

Per tensor head, all terms must share the same slot vbundle structure (variance
pattern); see [`_slot_structure`](@ref).

The zero element is `TensorComponentSum([])` — use [`is_zero`](@ref), not `== 0`.
"""
struct TensorComponentSum{T <: TensorComponentTerm} <: AbstractTensorComponentExpr
    terms::Vector{T}

    function TensorComponentSum(raw_terms::AbstractVector)
        if isempty(raw_terms)
            return new{TensorComponentTerm{Int, TensorComponent}}(
                TensorComponentTerm{Int, TensorComponent}[]
            )
        end
        terms = Vector{TensorComponentTerm}(raw_terms)
        _validate_sum_slot_structure!(terms)
        final_terms = _merge_terms(terms)
        if isempty(final_terms)
            return new{TensorComponentTerm{Int, TensorComponent}}(
                TensorComponentTerm{Int, TensorComponent}[]
            )
        end
        TT = eltype(final_terms)
        return new{TT}(final_terms)
    end
end

"""
    is_zero(s::TensorComponentSum) -> Bool

`true` when `s` is the zero element (`TensorComponentSum([])`).
Not the same as scalar `0`.
"""
is_zero(s::TensorComponentSum) = isempty(s.terms)

terms_of(s::TensorComponentSum) = s.terms


# =========================================
# 3.  Slot structure validation and merge
# =========================================

"""
    _slot_structure(comp::TensorComponent) -> Tuple{Symbol, Varargs{Symbol}}

Per-slot vbundle list encoding the variance pattern of `comp`.
"""
function _slot_structure(comp::TensorComponent)
    n = length(comp.indices)
    return Tuple(idx.vbundle for idx in comp.indices)
end

"""
    _validate_sum_slot_structure!(terms)

For each tensor head, require identical slot structure across all terms.
"""
function _validate_sum_slot_structure!(terms::Vector{<:TensorComponentTerm})
    sigs = Dict{Any, Tuple{Vararg{Symbol}}}()
    for t in terms
        body = t.body
        body isa TensorComponent || continue
        T = body.tensor
        sig = _slot_structure(body)
        if haskey(sigs, T)
            sigs[T] == sig ||
                throw(ArgumentError(
                    "TensorComponentSum: cannot add terms with tensor " *
                    "$(print_as(T)) using incompatible slot structures " *
                    "$(sigs[T]) and $(sig). " *
                    "Raise or lower indices explicitly before summing."
                ))
        else
            sigs[T] = sig
        end
    end
    return nothing
end

_collect_terms(t::TensorComponentTerm) = [t]
_collect_terms(s::TensorComponentSum) = collect(s.terms)

"""
    _merge_terms(raw_terms) -> Vector{TensorComponentTerm}

Merge by `body` using `Dict` keyed by body (`==` authoritative on collision).
Drop zero coeffs.
"""
function _merge_terms(raw_terms::Vector{<:TensorComponentTerm})
    isempty(raw_terms) && return TensorComponentTerm[]
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
    out = TensorComponentTerm{C, B}[]
    for (b, c) in merged
        is_scalar_zero(c) && continue
        push!(out, TensorComponentTerm(c, b))
    end
    return out
end


# =========================================
# 4.  Addition
# =========================================

Base.:+(a::TensorComponentTerm, b::TensorComponentTerm) =
    TensorComponentSum(TensorComponentTerm[a, b])
Base.:+(a::TensorComponentTerm, b::TensorComponentSum) =
    TensorComponentSum(vcat(_collect_terms(a), _collect_terms(b)))
Base.:+(a::TensorComponentSum, b::TensorComponentTerm) =
    TensorComponentSum(vcat(_collect_terms(a), _collect_terms(b)))
Base.:+(a::TensorComponentSum, b::TensorComponentSum) =
    TensorComponentSum(vcat(_collect_terms(a), _collect_terms(b)))

function Base.:+(::ScalarLike, ::AbstractTensorComponentExpr)
    throw(ArgumentError(
        "Cannot add scalar coefficient to tensor expression without a body. " *
        "Use TensorComponentTerm(coeff, body) or coeff * body."
    ))
end

function Base.:+(::AbstractTensorComponentExpr, ::ScalarLike)
    throw(ArgumentError(
        "Cannot add tensor expression to scalar coefficient without a body. " *
        "Use TensorComponentTerm(coeff, body) or coeff * body."
    ))
end

# Syntactic sugar: auto-promote components to terms
Base.:+(a::TensorComponent, b::TensorComponent) = term(a) + term(b)
Base.:+(a::TensorComponent, b::TensorComponentTerm) = term(a) + b
Base.:+(a::TensorComponentTerm, b::TensorComponent) = a + term(b)
Base.:+(a::TensorComponent, b::TensorComponentSum) = term(a) + b
Base.:+(a::TensorComponentSum, b::TensorComponent) = a + term(b)

Base.:-(a::TensorComponent) = -term(a)
Base.:-(a::AbstractTensorComponentExpr, b::AbstractTensorComponentExpr) = a + (-b)
Base.:-(a::TensorComponent, b::TensorComponent) = term(a) + (-term(b))
Base.:-(a::TensorComponent, b::AbstractTensorComponentExpr) = term(a) + (-b)
Base.:-(a::AbstractTensorComponentExpr, b::TensorComponent) = a + (-term(b))


# =========================================
# 5.  Scalar multiplication
# =========================================

Base.:*(c::ScalarLike, comp::TensorComponent) = TensorComponentTerm(c, comp)
Base.:*(comp::TensorComponent, c::ScalarLike) = TensorComponentTerm(c, comp)

Base.:*(c::ScalarLike, t::TensorComponentTerm) =
    TensorComponentTerm(scalar_mul(c, t.coeff), t.body)
Base.:*(t::TensorComponentTerm, c::ScalarLike) =
    TensorComponentTerm(scalar_mul(t.coeff, c), t.body)

Base.:*(c::ScalarLike, s::TensorComponentSum) =
    TensorComponentSum([
        TensorComponentTerm(scalar_mul(c, t.coeff), t.body) for t in s.terms
    ])
Base.:*(s::TensorComponentSum, c::ScalarLike) =
    TensorComponentSum([
        TensorComponentTerm(scalar_mul(c, t.coeff), t.body) for t in s.terms
    ])

function Base.:*(::TensorComponentTerm, ::TensorComponentTerm)
    throw(ArgumentError(
        "Multiplication of multiple tensor components requires a `TensorProduct` " *
        "layer, which is out of scope for this version."
    ))
end


# =========================================
# 6.  Unary minus
# =========================================

Base.:-(t::TensorComponentTerm) =
    TensorComponentTerm(scalar_mul(-1, t.coeff), t.body)
Base.:-(s::TensorComponentSum) =
    TensorComponentSum([
        TensorComponentTerm(scalar_mul(-1, t.coeff), t.body) for t in s.terms
    ])


# =========================================
# 7.  Display (MIME only)
# =========================================

function _format_term_plain(t::TensorComponentTerm)
    c = t.coeff
    if c == 1 || c == 1.0
        return string(t.body)
    end
    return "$(c) * $(t.body)"
end

function _format_term_latex(t::TensorComponentTerm)
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

function Base.show(io::IO, ::MIME"text/plain", t::TensorComponentTerm)
    print(io, _format_term_plain(t))
end

function Base.show(io::IO, ::MIME"text/plain", s::TensorComponentSum)
    if is_zero(s)
        print(io, "TensorComponentSum([])")
        return
    end
    parts = map(_format_term_plain, s.terms)
    print(io, join(parts, " + "))
end

function Base.show(io::IO, ::MIME"text/latex", t::TensorComponentTerm)
    print(io, "\$", _format_term_latex(t), "\$")
end

function Base.show(io::IO, ::MIME"text/latex", s::TensorComponentSum)
    if is_zero(s)
        print(io, "\$0_{\\text{tensor}}\$")
        return
    end
    parts = map(_format_term_latex, s.terms)
    print(io, "\$", join(parts, " + "), "\$")
end

function Base.show(io::IO, ::MIME"text/html", t::TensorComponentTerm)
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

function Base.show(io::IO, ::MIME"text/html", s::TensorComponentSum)
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

export AbstractTensorComponentExpr, TensorComponentTerm, TensorComponentSum
export term, coeff_of, body_of, terms_of, is_zero
