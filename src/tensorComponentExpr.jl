# =========================================
# tensorComponentExpr.jl — SymbolicTensors.jl
#
# Tensor algebra AST: TensorComponentTerm (coeff × body) and TensorComponentSum.
#
# Algebraic model: free module ⊕ (ScalarLike ⊗ body),
# body ∈ {[`TensorComponent`](@ref), [`TensorComponentProduct`](@ref)}.
#
# Depends on: scalar.jl, tensorComponents.jl
# Future: contractions.jl (index contraction)
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
- `body`  : [`TensorComponent`](@ref) or [`TensorComponentProduct`](@ref)
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
# 2.  TensorComponentProduct (geometric product body)
# =========================================

"""
    TensorComponentProduct

Canonical sorted product of two or more [`TensorComponent`](@ref) factors.
Construct via [`_make_product`](@ref), not the inner constructor directly.
"""
struct TensorComponentProduct
    factors::Vector{TensorComponent}
end

function TensorComponentProduct(factors::TensorComponent...)
    v = collect(factors)
    sort!(v; lt=is_canonical_less)
    return TensorComponentProduct(v)
end

_product_sorted(v::Vector{TensorComponent}) =
    TensorComponentProduct(v)

factors_of(p::TensorComponentProduct) = p.factors

function _lex_less_indices(a::Vector{AbstractIndex}, b::Vector{AbstractIndex})
    for (ia, ib) in zip(a, b)
        ia == ib && continue
        return isless(ia, ib)
    end
    return length(a) < length(b)
end

"""
    is_canonical_less(a, b) -> Bool

Total order for sorting product factors (commutative canonical form).
Compares [`tensor_id`](@ref) of heads, then lexicographic index order.
"""
function is_canonical_less(a::TensorComponent, b::TensorComponent)
    ka = tensor_id(a.tensor)
    kb = tensor_id(b.tensor)
    ka != kb && return ka < kb
    return _lex_less_indices(a.indices, b.indices)
end

function is_canonical_less(a::TensorComponentProduct, b::TensorComponentProduct)
    for (fa, fb) in zip(a.factors, b.factors)
        fa == fb && continue
        return is_canonical_less(fa, fb)
    end
    return length(a.factors) < length(b.factors)
end

# Fallbacks just in case a sum mixes single components and products
is_canonical_less(a::TensorComponent, b::TensorComponentProduct) = true
is_canonical_less(a::TensorComponentProduct, b::TensorComponent) = false

Base.:(==)(p::TensorComponentProduct, q::TensorComponentProduct) = p.factors == q.factors
Base.hash(p::TensorComponentProduct, h::UInt) = hash(p.factors, h)

_collect_product_factors(c::TensorComponent) = [c]
_collect_product_factors(p::TensorComponentProduct) = collect(p.factors)

"""
    _make_product(factors...) -> Union{TensorComponent, TensorComponentProduct}

Flatten nested products; return a single component or a canonical product.
"""
function _make_product(factors::TensorComponent...)
    flat = TensorComponent[]
    for x in factors
        if x isa TensorComponent
            push!(flat, x)
        elseif x isa TensorComponentProduct
            append!(flat, collect(x.factors))
        else
            throw(ArgumentError("_make_product: expected TensorComponent factors."))
        end
    end
    length(flat) == 0 && throw(ArgumentError("empty product"))
    length(flat) == 1 && return flat[1]
    return TensorComponentProduct(flat...)
end

"""
    _multiply_bodies(a, b)

Geometric product of two term bodies (component and/or product).
"""
function _multiply_bodies(
    a::TensorComponent,
    b::TensorComponent,
)
    if is_canonical_less(a, b)
        return _product_sorted(TensorComponent[a, b])
    else
        return _product_sorted(TensorComponent[b, a])
    end
end
function _multiply_bodies(
    a::TensorComponent,
    b::TensorComponentProduct,
)
    v = _insert_sorted(b.factors, a)
    return _product_sorted(v)
end
function _multiply_bodies(
    a::TensorComponentProduct,
    b::TensorComponent,
)
    v = _insert_sorted(a.factors, b)
    return _product_sorted(v)
end
function _multiply_bodies(
    a::TensorComponentProduct,
    b::TensorComponentProduct,
)
    v = _merge_sorted_factors(a.factors, b.factors)
    return _product_sorted(v)
end

function _insert_sorted(
    factors::Vector{TensorComponent},
    x::TensorComponent,
)
    out = Vector{TensorComponent}(undef, length(factors)+1)

    inserted = false
    j = 1

    @inbounds for i in eachindex(factors)
        fi = factors[i]

        if !inserted && is_canonical_less(x, fi)
            out[j] = x
            j += 1
            inserted = true
        end

        out[j] = fi
        j += 1
    end

    if !inserted
        out[end] = x
    end

    return out
end

function _merge_sorted_factors(
    a::Vector{TensorComponent},
    b::Vector{TensorComponent},
)
    na = length(a)
    nb = length(b)

    out = Vector{TensorComponent}(undef, na + nb)

    ia = 1
    ib = 1
    k = 1

    @inbounds while ia <= na && ib <= nb
        if is_canonical_less(a[ia], b[ib])
            out[k] = a[ia]
            ia += 1
        else
            out[k] = b[ib]
            ib += 1
        end
        k += 1
    end

    @inbounds while ia <= na
        out[k] = a[ia]
        ia += 1
        k += 1
    end

    @inbounds while ib <= nb
        out[k] = b[ib]
        ib += 1
        k += 1
    end

    return out
end

term(body::TensorComponentProduct) = TensorComponentTerm(one_scalar(1), body)

# Mixed Term / Geometric multiplication sugar
Base.:*(a::TensorComponentTerm, b::TensorComponent) = a * term(b)
Base.:*(a::TensorComponent, b::TensorComponentTerm) = term(a) * b

Base.:*(a::TensorComponentTerm, p::TensorComponentProduct) = a * term(p)
Base.:*(p::TensorComponentProduct, b::TensorComponentTerm) = term(p) * b


# =========================================
# 3.  TensorComponentSum — sole merge bottleneck
# =========================================

"""
    TensorComponentSum{T<:TensorComponentTerm}

Finite sum of [`TensorComponentTerm`](@ref)s. Normalization (merge by body,
drop zero coeffs) runs only in the inner constructor.

Summand slot homogeneity is **not** checked here; call [`validate`](@ref) explicitly.

The zero element is `TensorComponentSum([])` — use [`is_zero`](@ref), not `== 0`.
"""
struct TensorComponentSum{T <: TensorComponentTerm} <: AbstractTensorComponentExpr
    terms::Vector{T}

    function TensorComponentSum(raw_terms::AbstractVector{T}) where {T <: TensorComponentTerm}
        isempty(raw_terms) && return new{T}(T[])
        final_terms = _merge_terms(raw_terms)
        isempty(final_terms) && return new{T}(T[])
        return new{eltype(final_terms)}(final_terms)
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
# 4.  Slot structure validation and merge
# =========================================

_slot_structure(comp::TensorComponent) =
    Tuple(idx.vbundle for idx in comp.indices)

_slot_structure(prod::TensorComponentProduct) =
    Tuple(idx.vbundle for comp in prod.factors for idx in comp.indices)

_sum_body_head(comp::TensorComponent) = comp.tensor
_sum_body_head(prod::TensorComponentProduct) = Tuple(c.tensor for c in prod.factors)

"""
    _validate_sum_slot_structure!(terms)

For each tensor head (or product head tuple), require identical slot structure.
Used by [`validate`](@ref); not called from the [`TensorComponentSum`](@ref) constructor.
"""
function _validate_sum_slot_structure!(terms::Vector{<:TensorComponentTerm})
    sigs = Dict{Any, Tuple{Vararg{Symbol}}}()
    for t in terms
        body = t.body
        (body isa TensorComponent || body isa TensorComponentProduct) || continue
        head = _sum_body_head(body)
        sig = _slot_structure(body)
        if haskey(sigs, head)
            sigs[head] == sig ||
                throw(ArgumentError(
                    "TensorComponentSum: incompatible slot structures " *
                    "$(sigs[head]) and $(sig) for head $(head). " *
                    "Raise or lower indices explicitly before summing."
                ))
        else
            sigs[head] = sig
        end
    end
    return nothing
end

"""
    validate(expr::AbstractTensorComponentExpr)

Check summand slot homogeneity in [`TensorComponentSum`](@ref)s: for each tensor
head, all terms must share the same slot vbundle structure (variance pattern).

Returns `expr` unchanged when valid. Not invoked automatically by `+` or `sum`.
"""
validate(t::TensorComponentTerm) = t

function validate(s::TensorComponentSum)
    _validate_sum_slot_structure!(terms_of(s))
    return s
end

_collect_terms(t::TensorComponentTerm) = [t]
_collect_terms(s::TensorComponentSum)  = s.terms

"""
    _merge_terms(raw_terms) -> Vector{TensorComponentTerm}

Merge by `body` using `Dict` keyed by body (`==` authoritative on collision).
Drop zero coeffs.

Avoids slicing `raw_terms[2:end]` (which allocates a copy); iterates by index
instead. `sizehint!` prevents Dict rehashing on large inputs.
"""
function _merge_terms(raw_terms::AbstractVector{T}) where {T <: TensorComponentTerm}
    isempty(raw_terms) && return T[]
    
    # 1. Sort a copy of the array using our fast rules
    sorted = sort(raw_terms, lt = (x, y) -> is_canonical_less(x.body, y.body))
    
    # 2. Linear Scan
    out = T[]
    sizehint!(out, 8) # Pre-allocate a small buffer
    
    # Use firstindex instead of 1
    curr_body = sorted[firstindex(sorted)].body
    curr_coeff = sorted[firstindex(sorted)].coeff
    
    # Use lastindex instead of length
    @inbounds for i in (firstindex(sorted) + 1):lastindex(sorted)
        t = sorted[i]
        if t.body == curr_body
            # Same body -> accumulate coefficient
            curr_coeff = scalar_add(curr_coeff, t.coeff)
        else
            # New body -> push previous (if non-zero) and reset
            if !is_scalar_zero(curr_coeff)
                push!(out, TensorComponentTerm(curr_coeff, curr_body))
            end
            curr_body = t.body
            curr_coeff = t.coeff
        end
    end
    
    # Push the final accumulated term
    if !is_scalar_zero(curr_coeff)
        push!(out, TensorComponentTerm(curr_coeff, curr_body))
    end
    
    return out
end

# =========================================
# 5.  Addition
# =========================================

Base.:+(a::TensorComponentTerm{CA,BA}, b::TensorComponentTerm{CB,BB}) where {CA,BA,CB,BB} =
    TensorComponentSum([a, b])

Base.:+(a::TensorComponentTerm{C,B}, b::TensorComponentSum{TB}) where {C,B,TB} =
    TensorComponentSum(vcat([a], b.terms))

Base.:+(a::TensorComponentSum{TA}, b::TensorComponentTerm{C,B}) where {TA,C,B} =
    TensorComponentSum(vcat(a.terms, [b]))

Base.:+(a::TensorComponentSum{TA}, b::TensorComponentSum{TB}) where {TA,TB} =
    TensorComponentSum(vcat(a.terms, b.terms))

"""
    Base.sum(terms::AbstractArray{<:TensorComponentTerm})

Bulk-add terms in a single [`TensorComponentSum`](@ref) merge pass.
Prefer this over chained `+` / the default `sum` fold for long vectors.
"""
Base.sum(terms::AbstractArray{<:TensorComponentTerm}) = TensorComponentSum(terms)

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
Base.:+(a::TensorComponentProduct, b::TensorComponentProduct) = term(a) + term(b)
Base.:+(a::TensorComponentProduct, b::TensorComponent) = term(a) + term(b)
Base.:+(a::TensorComponent, b::TensorComponentProduct) = term(a) + term(b)
Base.:+(a::TensorComponentProduct, b::TensorComponentTerm) = term(a) + b
Base.:+(a::TensorComponentTerm, b::TensorComponentProduct) = a + term(b)
Base.:+(a::TensorComponentProduct, b::TensorComponentSum) = term(a) + b
Base.:+(a::TensorComponentSum, b::TensorComponentProduct) = a + term(b)

Base.:-(a::TensorComponent) = -term(a)
Base.:-(p::TensorComponentProduct) = -term(p)
Base.:-(p::TensorComponentProduct, b::TensorComponentProduct) = term(p) + (-term(b))
Base.:-(p::TensorComponentProduct, b::AbstractTensorComponentExpr) = term(p) + (-b)
Base.:-(a::AbstractTensorComponentExpr, p::TensorComponentProduct) = a + (-term(p))
Base.:-(a::AbstractTensorComponentExpr, b::AbstractTensorComponentExpr) = a + (-b)
Base.:-(a::TensorComponent, b::TensorComponent) = term(a) + (-term(b))
Base.:-(a::TensorComponent, b::AbstractTensorComponentExpr) = term(a) + (-b)
Base.:-(a::AbstractTensorComponentExpr, b::TensorComponent) = a + (-term(b))


# =========================================
# 6.  Scalar and geometric multiplication
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

# Geometric product on components / products
Base.:*(a::TensorComponent, b::TensorComponent) = _make_product(a, b)
Base.:*(a::TensorComponentProduct, b::TensorComponent) =
    _make_product(a.factors..., b)
Base.:*(a::TensorComponent, b::TensorComponentProduct) =
    _make_product(a, b.factors...)
Base.:*(a::TensorComponentProduct, b::TensorComponentProduct) =
    _make_product(a.factors..., b.factors...)

# Term × term (coefficients on scalars; bodies via _multiply_bodies)
Base.:*(a::TensorComponentTerm, b::TensorComponentTerm) =
    TensorComponentTerm(
        scalar_mul(a.coeff, b.coeff),
        _multiply_bodies(a.body, b.body),
    )

# Distributivity
Base.:*(a::TensorComponentTerm, s::TensorComponentSum) =
    TensorComponentSum([a * t for t in s.terms])
Base.:*(s::TensorComponentSum, b::TensorComponentTerm) =
    TensorComponentSum([t * b for t in s.terms])
Base.:*(a::TensorComponent, s::TensorComponentSum) = term(a) * s
Base.:*(s::TensorComponentSum, b::TensorComponent) = s * term(b)
Base.:*(p::TensorComponentProduct, s::TensorComponentSum) = term(p) * s
Base.:*(s::TensorComponentSum, p::TensorComponentProduct) = s * term(p)
Base.:*(s1::TensorComponentSum, s2::TensorComponentSum) =
    TensorComponentSum([t1 * t2 for t1 in s1.terms for t2 in s2.terms])


# =========================================
# 7.  Unary minus
# =========================================

Base.:-(t::TensorComponentTerm) =
    TensorComponentTerm(scalar_mul(-1, t.coeff), t.body)
Base.:-(s::TensorComponentSum) =
    TensorComponentSum([
        TensorComponentTerm(scalar_mul(-1, t.coeff), t.body) for t in s.terms
    ])


# =========================================
# 8.  Display (MIME only)
# =========================================

function _format_component_plain(c::TensorComponent)
    idx_strs = map(c.indices) do idx
        is_down(idx) ? "-$(idx.symbol)" : "$(idx.symbol)"
    end
    "$(print_as(c.tensor))[$(join(idx_strs, ", "))]"
end


function _format_body_plain(body)
    if body isa TensorComponent
        return _format_component_plain(body)
    else
        return _format_product_plain(body)
    end
end

function _format_body_latex(body)
    if body isa TensorComponent
        return _format_component_latex(body)
    else
        return _format_product_latex(body)
    end
end


function _format_component_latex(c::TensorComponent)
    _format_latex(c)
end

function _format_component_html(c::TensorComponent)
    _format_html(c)
end



function _format_body_html(body)
    if body isa TensorComponent
        return _format_component_html(body)
    else
        return _format_product_html(body)
    end
end



function _format_term_plain(t::TensorComponentTerm)
    c = t.coeff
    if c == 1 || c == 1.0
        return _format_body_plain(t.body)
    elseif c == -1 || c == -1.0
        return "-$(_format_body_plain(t.body))"
    end
    return "$(c) * $(_format_body_plain(t.body))"
end

function _format_term_latex(t::TensorComponentTerm)
    c = t.coeff
    body = t.body
    if c == 1 || c == 1.0
        return _format_body_latex(body)
    end
    return "$(c)\\,\\($(_format_body_latex(body))\\)"
end

function Base.show(io::IO, ::MIME"text/plain", t::TensorComponentTerm)
    print(io, _format_term_plain(t))
end

function Base.show(io::IO, ::MIME"text/html", t::TensorComponentTerm)
    c = t.coeff
    body = t.body
    if c == 1 || c == 1.0
        print(io, _format_body_html(body))
        return
    elseif c == -1 || c == -1.0
        print(io, "<span>-</span>", _format_body_html(body))
        return
    end
    print(io, "<span>", c, " · </span>", _format_body_html(body))
end

function Base.show(io::IO, ::MIME"text/latex", t::TensorComponentTerm)
    c = t.coeff
    body = t.body
    if c == 1 || c == 1.0
        return _format_body_latex(body)
    elseif c == -1 || c == -1.0
        return "-\\left($(_format_body_latex(body))\\right)"
    end
    return "$(c)\\,\\left($(_format_body_latex(body))\\right)"
end

function Base.show(io::IO, ::MIME"text/plain", s::TensorComponentSum)
    if is_zero(s)
        print(io, "TensorComponentSum([])")
        return
    end
    parts = map(_format_term_plain, s.terms)
    print(io, join(parts, " + "))
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

function Base.show(io::IO, ::MIME"text/latex", s::TensorComponentSum)
    if is_zero(s)
        print(io, "\$0_{\\text{tensor}}\$")
        return
    end
    parts = map(_format_term_latex, s.terms)
    print(io, "\$", join(parts, " + "), "\$")
end

function _format_product_plain(p::TensorComponentProduct)
    join((_format_component_plain(f) for f in p.factors), " * ")
end

function _format_product_html(p::TensorComponentProduct)
    parts = _format_component_html.(p.factors)
    join(parts, " · ")
end

function _format_product_latex(p::TensorComponentProduct)
    join((_format_component_latex(f) for f in p.factors), "\\,")
end

function Base.show(io::IO, ::MIME"text/plain", p::TensorComponentProduct)
    print(io, _format_product_plain(p))
end

function Base.show(io::IO, ::MIME"text/latex", p::TensorComponentProduct)
    print(io, "\$", _format_product_latex(p), "\$")
end

function Base.show(io::IO, ::MIME"text/html", p::TensorComponentProduct)
    print(io, _format_product_html(p))
end


# =========================================
# Exports
# =========================================

export AbstractTensorComponentExpr, TensorComponentTerm, TensorComponentSum
export TensorComponentProduct, factors_of
export term, coeff_of, body_of, terms_of, is_zero, validate
export is_canonical_less
