# =========================================
# tensorExpressions.jl — SymbolicTensors.jl
#
# A TensorExpression is a Tensor (schema) applied to a specific list of
# AbstractIndex objects. It is the REPL/notebook object you interact with:
#
#   F[down(a1), down(a2)]   — explicit construction
#   F[-a1, -a2]             — sugar: -a1 calls flip on AbstractIndex
#   F[a1, a2]               — also valid: contravariant expression
#   g[a1, a2]               — valid: contravariant metric (raised by implicit g)
#
# Design principle: slot variance (stored on Tensor.slots) records the
# *canonical* index placement declared at @def_tensor time. It is NOT
# enforced at TensorExpression construction. Raising and lowering indices
# is a separate algebraic operation (future raise_index / lower_index).
#
# What IS validated at construction:
#   - Arity: number of indices must equal T.rank.
#   - Manifold membership: each index must belong to the correct manifold.
#     Concretely, index_home_vbundle(idx.symbol) must equal T's tangent bundle.
#
# What is NOT validated:
#   - Variance (up/down) of each index against the slot declaration.
#     T[-a1,-a2] with metric g allows g[a1,a2], T[a1,-a2], etc.
#
# The struct is lazy and inert: no contraction, no symmetry reduction
# happens at construction time. Algebra (*,+) and contraction are
# implemented in future files (tensorAlgebra.jl, contractions.jl).
#
# Depends on: indices.jl, manifolds.jl, permutations.jl, tensors.jl, metrics.jl
# =========================================


# =========================================
# 1.  TensorExpression struct
# =========================================

"""
    TensorExpression

A [`Tensor`](@ref) (definition-level schema) applied to a concrete list of
[`AbstractIndex`](@ref) objects, representing one occurrence in an algebraic
expression.

Constructed via `getindex` on a [`Tensor`](@ref):

    F[down(a1), down(a2)]   # explicit AbstractIndex arguments
    F[-a1, -a2]             # sugar: -a1 = flip(a1) via Base.:- on AbstractIndex
    F[a1, -a2]              # mixed: a1 contravariant, -a2 covariant
    g[a1, a2]               # valid: contravariant metric expression

**Validation at construction time:**
- `length(idxs) == T.rank`
- *Manifold membership*: each index's home tangent bundle must match `T.manifold`

**Not validated:**
- *Variance* (up vs down) against `T.slots`. The canonical slot structure
  is metadata for index raising/lowering, not a construction gate.

The expression is **lazy and inert**: no contraction, canonicalization, or
symmetry reduction is performed at construction time.

### Fields

- `tensor`  : the [`Tensor`](@ref) this expression refers to
- `indices` : the concrete index list for this occurrence, one per slot
"""
struct TensorExpression
    tensor::Tensor
    indices::Vector{AbstractIndex}
end


# =========================================
# 2.  Internal argument parser
# =========================================

# Accepts the forms a slot argument can take at runtime:
#   AbstractIndex  — used directly (includes -a1 / +a1 after unary ops)
#   Symbol       — treated as contravariant (up); must be registered
# Note: IndexSymbol no longer exists; index variables in scope are CoordinateIndex or BasisIndex.
function _parse_index_arg(arg)::AbstractIndex
    if arg isa AbstractIndex
        return arg
    elseif arg isa Symbol
        is_index_registered(arg) ||
            error(
                "TensorExpression: index :$arg is not registered. " *
                "Call @def_manifold or @add_indices first."
            )
        return up(arg)
    else
        error(
            "TensorExpression: cannot interpret slot argument $(repr(arg)) " *
            "of type $(typeof(arg)). " *
            "Use -a1 (covariant) or a1 (contravariant) index values."
        )
    end
end


# ======================================================
# 3.  Base.getindex — T[a1, -a2, ...] → TensorExpression
# ======================================================

"""
    Base.getindex(T::Tensor, idxs...) -> TensorExpression

Construct a [`TensorExpression`](@ref) by applying `T` to the given indices.

Accepted argument types per slot:
- [`AbstractIndex`](@ref)       — used directly (contravariant or covariant)
- `-`[`AbstractIndex`](@ref)    — covariant via `flip`; unary `-` on `AbstractIndex`
- `+`[`AbstractIndex`](@ref)    — contravariant (identity); unary `+` on `AbstractIndex`

**Validated:**
1. **Arity** — `length(idxs) == T.rank`
2. **Manifold membership** — each index's home tangent bundle matches `T.manifold`

**Not validated:**
- Variance against `T.slots`. Any up/down combination is accepted.
  `g[a1, a2]`, `T[a1, -a2]`, `T[-a1, -a2]` are all valid expressions.

# Examples
~~~julia
@def_manifold M 4 [a1, a2, a3, a4] [A1, A2, A3, A4]
@def_metric g M
@def_tensor F[-a1, -a2] M symmetries=[antisymmetric(2)]

g[-a1, -a2]              # covariant metric (canonical form)
g[a1, a2]                # contravariant metric (raised) — valid
g[a1, -a2]               # mixed — valid
F[-a1, -a2]              # covariant F (canonical form)
F[a1, a2]                # contravariant F — valid
F[-a1, -a2, -a1]         # error: rank mismatch
~~~
"""
function Base.getindex(T::Tensor, idxs...)
    n = length(idxs)

    n == T.rank ||
        error(
            "TensorExpression: tensor $(T.print_as) has rank $(T.rank) " *
            "but $n index argument(s) were given."
        )

    # Step 1: parse each argument to an AbstractIndex.
    ti = Vector{AbstractIndex}(undef, n)
    for i in 1:n
        ti[i] = _parse_index_arg(idxs[i])
    end

    # Step 2: retrieve manifold metadata for validation.
    haskey(_MANIFOLDS, T.manifold) ||
        error(
            "TensorExpression: tensor $(T.print_as) references " *
            "unregistered manifold :$(T.manifold)."
        )
    tb = _MANIFOLDS[T.manifold].tangent_bundle   # e.g. :tangentM

    # Step 3: manifold membership validation only.
    # Variance is NOT checked — any up/down combination is valid.
    for i in 1:n
        idx = ti[i]

        is_index_registered(idx.symbol) ||
            error(
                "TensorExpression: index :$(idx.symbol) is not registered."
            )

        home = index_home_vbundle(idx.symbol)
        home == tb ||
            error(
                "TensorExpression: index :$(idx.symbol) has home bundle " *
                ":$home, expected :$tb (manifold $(T.manifold))."
            )
    end

    TensorExpression(T, ti)
end


# =========================================
# 4.  Accessors
# =========================================

"""Return the [`Tensor`](@ref) schema of a `TensorExpression`."""
tensor_of(e::TensorExpression)  = e.tensor

"""Return the concrete index list of a `TensorExpression`."""
indices_of_tensor(e::TensorExpression) = e.indices

"""
    rank_of(e::TensorExpression) -> Int

Number of slots of the expression. Dispatches alongside `rank_of(::Tensor)`.
"""
rank_of(e::TensorExpression) = length(e.indices)

"""
    canonical_slots(e::TensorExpression) -> Vector{Symbol}

Return the canonical slot vbundles from the underlying [`Tensor`](@ref) schema.
These record the index placement declared at `@def_tensor` time and are used
by `raise_index` / `lower_index` (future), not for construction validation.
"""
canonical_slots(e::TensorExpression) = e.tensor.slots

"""
    variance_matches_canonical(e::TensorExpression) -> Bool

Return `true` if every index in `e` matches the canonical slot variance
declared in `e.tensor.slots`. Useful for diagnostics and for triggering
automatic index raising/lowering in algebraic simplification.
"""
function variance_matches_canonical(e::TensorExpression)
    for (idx, slot_vb) in zip(e.indices, e.tensor.slots)
        idx.vbundle == slot_vb || return false
    end
    return true
end


# =========================================
# 5.  Equality and hashing
# =========================================

Base.:(==)(a::TensorExpression, b::TensorExpression) =
    a.tensor === b.tensor && a.indices == b.indices

Base.hash(e::TensorExpression, h::UInt) =
    hash((objectid(e.tensor), e.indices), h)


# =========================================
# 6.  Display helpers
# =========================================

# Unicode sub/superscript maps for Latin letters only.
# Characters not in the table are kept as-is (normal height fallback).
const _CHAR_TO_SUB = Dict{Char,Char}(
    'a'=>'ₐ','e'=>'ₑ','h'=>'ₕ','i'=>'ᵢ','j'=>'ⱼ',
    'k'=>'ₖ','l'=>'ₗ','m'=>'ₘ','n'=>'ₙ','o'=>'ₒ',
    'p'=>'ₚ','r'=>'ᵣ','s'=>'ₛ','t'=>'ₜ','u'=>'ᵤ',
    'v'=>'ᵥ','x'=>'ₓ',
)

const _CHAR_TO_SUP = Dict{Char,Char}(
    'a'=>'ᵃ','b'=>'ᵇ','c'=>'ᶜ','d'=>'ᵈ','e'=>'ᵉ',
    'f'=>'ᶠ','g'=>'ᵍ','h'=>'ʰ','i'=>'ⁱ','j'=>'ʲ',
    'k'=>'ᵏ','l'=>'ˡ','m'=>'ᵐ','n'=>'ⁿ','o'=>'ᵒ',
    'p'=>'ᵖ','r'=>'ʳ','s'=>'ˢ','t'=>'ᵗ','u'=>'ᵘ',
    'v'=>'ᵛ','w'=>'ʷ','x'=>'ˣ','y'=>'ʸ','z'=>'ᶻ',
)

function _map_chars(sym::Symbol, table::Dict{Char,Char})
    join(get(table, c, c) for c in string(sym))
end

function _is_covariant_idx(idx::AbstractIndex)
    haskey(_VBUNDLES, idx.vbundle) && is_down(idx)
end

"""
    _group_index_runs(indices) -> Vector{Tuple{Bool, Vector{Symbol}}}

Group consecutive indices of the same variance into runs.
Each element is `(is_covariant, [sym1, sym2, ...])`.

    [-a1, -a2, a3]  →  [(true,[:a1,:a2]), (false,[:a3])]
"""
function _group_index_runs(indices::Vector{AbstractIndex})
    isempty(indices) && return Tuple{Bool,Vector{Symbol}}[]
    runs = Tuple{Bool,Vector{Symbol}}[]
    cur_cov = _is_covariant_idx(indices[1])
    cur_syms = [indices[1].symbol]
    for idx in indices[2:end]
        cov = _is_covariant_idx(idx)
        if cov == cur_cov
            push!(cur_syms, idx.symbol)
        else
            push!(runs, (cur_cov, cur_syms))
            cur_cov  = cov
            cur_syms = [idx.symbol]
        end
    end
    push!(runs, (cur_cov, cur_syms))
    runs
end


# ==============================================
# 7.  Formatted output — three shared formatters
# ==============================================

"""
    _format_latex(e::TensorExpression) -> String

Produce a LaTeX math-mode string (without surrounding `\$`).

Examples:
    g[-a1, -a2]  → "g_{a_{1} a_{2}}"
    g[a1, a2]    → "g^{a_{1} a_{2}}"
    T[a1, -a2]   → "T^{a_{1}}_{a_{2}}"
"""
function _format_latex(e::TensorExpression)
    function latex_sym(sym::Symbol)
        s = string(sym)
        m = match(r"^([^\d]*)(\d+)$", s)
        m === nothing ? s : "$(m[1])_{$(m[2])}"
    end

    runs = _group_index_runs(e.indices)
    buf  = string(e.tensor.print_as)
    for (is_cov, syms) in runs
        body = join(latex_sym(s) * " " for s in syms) |> rstrip
        buf *= is_cov ? "_{$body}" : "^{$body}"
    end
    buf
end

"""
    _format_html(e::TensorExpression) -> String
 
Produce an HTML string for Jupyter / Pluto display.
The tensor name is rendered as-is; covariant indices appear in `<sub>` tags
and contravariant indices in `<sup>` tags, with no additional styling.
"""
function _format_html(e::TensorExpression)
    runs = _group_index_runs(e.indices)
    buf  = string(e.tensor.print_as)
    for (is_cov, syms) in runs
        tag   = is_cov ? "sub" : "sup"
        inner = join(string.(syms), " ")
        buf  *= "<$tag>$inner</$tag>"
    end
    buf
end
 

# =========================================
# 8.  show methods
# =========================================
 
"""
    Base.show(io::IO, e::TensorExpression)
 
Plain-text / REPL display. Renders as `name[±idx1, ±idx2, ...]` where
covariant indices are prefixed with `-` and contravariant indices are bare.
 
    g[-a1, -a2]   →  g[-a1, -a2]
    g[a1, a2]     →  g[a1, a2]
    T[a1, -a2]    →  T[a1, -a2]
"""
function Base.show(io::IO, ::MIME"text/plain", e::TensorExpression)
    idx_strs = map(e.indices) do idx
        is_down(idx) ? "-$(idx.symbol)" : "$(idx.symbol)"
    end
    print(io, "$(e.tensor.print_as)[$(join(idx_strs, ", "))]")
end
 
"""
    Base.show(io::IO, ::MIME"text/latex", e::TensorExpression)
 
LaTeX display for IJulia / Jupyter notebooks.
"""
function Base.show(io::IO, ::MIME"text/latex", e::TensorExpression)
    print(io, "\$", _format_latex(e), "\$")
end
 
"""
    Base.show(io::IO, ::MIME"text/html", e::TensorExpression)
 
HTML display for Jupyter / Pluto notebooks.
"""
function Base.show(io::IO, ::MIME"text/html", e::TensorExpression)
    print(io, _format_html(e))
end
 



# =========================================
# Exports
# =========================================

export TensorExpression
export tensor_of, indices_of_tensor, canonical_slots, variance_matches_canonical
# rank_of: already exported from tensors.jl; the TensorExpression method
# is added here via multiple dispatch — no re-export needed.