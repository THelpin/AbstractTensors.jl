# =========================================
# tensorComponents.jl — SymbolicTensors.jl
#
# A TensorComponent is an AbstractTensor (schema) applied to a specific list of
# AbstractIndex objects. It is the REPL/notebook object you interact with:
#
#   F[down(a1), down(a2)]   — explicit construction
#   F[-a1, -a2]             — sugar: -a1 calls flip on AbstractIndex
#   F[a1, a2]               — also valid: contravariant expression
#   g[a1, a2]               — valid: contravariant metric (raised by implicit g)
#
# Design principle: slot variance (stored on Tensor.slots) records the
# *canonical* index placement declared at @def_tensor time. When the tensor
# has an associated metric, per-slot variance is not enforced at construction
# (the metric identifies V with V*). Without a metric, each index must lie
# exactly on the declared slot vbundle.
#
# What IS validated at construction:
#   - Arity: number of indices must equal T.rank.
#   - Manifold membership: each index's vbundle lies over T.manifold.
#   - Vbundle of reference: each index derives from T.vbundle.
#   - Per slot (no metric): idx.vbundle == T.slots[i] (exact canonical placement).
#   - Per slot (with metric): skipped — any up/down on T.vbundle is allowed.
#
# The struct is lazy and inert: no contraction, no symmetry reduction
# happens at construction time. Algebra (*,+) and contraction are
# implemented in future files (tensorAlgebra.jl, contractions.jl).
#
# Depends on: indices.jl, manifolds.jl, permutations.jl, tensors.jl, metrics.jl
# =========================================


# =========================================
# 1.  TensorComponent struct
# =========================================

"""
    TensorComponent

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
- *Manifold membership*: each index's `vbundle` belongs to `T.manifold`
- *Vbundle of reference*: `_vbundle_of_reference_of(idx.vbundle) == T.vbundle`
- *Per slot* (only if `T.metric === nothing`): `idx.vbundle == T.slots[i]`
  (exact match to canonical slot placement)

**With a metric** (`T.metric !== nothing`): per-slot checks are skipped;
`g[a1, a2]` is valid even when canonical slots are `[cotangentM, cotangentM]`.

The expression is **lazy and inert**: no contraction, canonicalization, or
symmetry reduction is performed at construction time.

### Fields

- `tensor`  : the [`AbstractTensor`](@ref) this expression refers to
- `indices` : the concrete index list for this occurrence, one per slot
"""
struct TensorComponent
    tensor::AbstractTensor
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
                "TensorComponent: index :$arg is not registered. " *
                "Call @def_manifold or @add_indices first."
            )
        return up(arg)
    else
        error(
            "TensorComponent: cannot interpret slot argument $(repr(arg)) " *
            "of type $(typeof(arg)). " *
            "Use -a1 (covariant) or a1 (contravariant) index values."
        )
    end
end


# ======================================================
# 3.  Base.getindex — T[a1, -a2, ...] → TensorComponent
# ======================================================

"""
    Base.getindex(T::Tensor, idxs...) -> TensorComponent

Construct a [`TensorComponent`](@ref) by applying `T` to the given indices.

Accepted argument types per slot:
- [`AbstractIndex`](@ref)       — used directly (contravariant or covariant)
- `-`[`AbstractIndex`](@ref)    — covariant via `flip`; unary `-` on `AbstractIndex`
- `+`[`AbstractIndex`](@ref)    — contravariant (identity); unary `+` on `AbstractIndex`

**Validated:**
1. **Arity** — `length(idxs) == T.rank`
2. **Manifold membership** — each index's vbundle lies over `T.manifold`
3. **Vbundle of reference** — each index derives from `T.vbundle`
4. **Per slot** — if `T.metric === nothing`, `idx.vbundle == T.slots[i]` exactly;
   if `T.metric !== nothing`, per-slot vbundle matching is skipped

# Examples
~~~julia
@def_manifold M 4 [a1, a2, a3, a4] [A1, A2, A3, A4]
@def_metric g tangentM
@def_tensor T [cotangentM, cotangentM]

g[-a1, -a2]              # covariant (canonical)
g[a1, a2]                # raised — valid (metric present)

@def_tensor F [cotangentM, cotangentM]   # no metric on M
F[-a1, -a2]              # valid
F[a1, a2]                # error: a1 is on tangentM, slot expects cotangentM

@def_vbundle E M 4 [B1, B2, B3, B4]
@def_tensor K [E, dualE]
K[B1, -B2]               # valid
K[-B1, B2]               # error: wrong slot vbundles
~~~
"""
function Base.getindex(T::Tensor, idxs...)
    n = length(idxs)

    n == T.rank ||
        error(
            "TensorComponent: tensor $(T.print_as) has rank $(T.rank) " *
            "but $n index argument(s) were given."
        )

    # Step 1: parse each argument to an AbstractIndex.
    ti = Vector{AbstractIndex}(undef, n)
    for i in 1:n
        ti[i] = _parse_index_arg(idxs[i])
    end

    # Step 2: validate tensor manifold is registered.
    haskey(_MANIFOLDS, T.manifold) ||
        error(
            "TensorComponent: tensor $(T.print_as) references " *
            "unregistered manifold :$(T.manifold)."
        )

    ref_vb = T.vbundle

    for i in 1:n
        idx = ti[i]

        is_index_registered(idx.symbol) ||
            error(
                "TensorComponent: index :$(idx.symbol) is not registered."
            )

        haskey(_VBUNDLES, idx.vbundle) ||
            error(
                "TensorComponent: index :$(idx.symbol) has unregistered " *
                "vbundle :$(idx.vbundle)."
            )
        _VBUNDLES[idx.vbundle].manifold == T.manifold ||
            error(
                "TensorComponent: index :$(idx.symbol) is on manifold " *
                ":$(_VBUNDLES[idx.vbundle].manifold), but tensor $(T.print_as) " *
                "is on :$(T.manifold)."
            )

        _vbundle_of_reference_of(idx.vbundle) == ref_vb ||
            error(
                "TensorComponent: index :$(idx.symbol) has vbundle of reference " *
                ":$(_vbundle_of_reference_of(idx.vbundle)), but tensor " *
                "$(T.print_as) has vbundle of reference :$ref_vb."
            )

        if T.metric === nothing
            slot_vb = T.slots[i]
            idx.vbundle == slot_vb ||
                error(
                    "TensorComponent: index :$(idx.symbol) is on vbundle " *
                    ":$(idx.vbundle), but slot $i of $(T.print_as) expects " *
                    ":$slot_vb (tensor has no metric for raising/lowering)."
                )
        end
    end

    TensorComponent(T, ti)
end


# =========================================
# 4.  Accessors
# =========================================

"""Return the [`AbstractTensor`](@ref) schema of a `TensorComponent`."""
tensor_of(e::TensorComponent)  = e.tensor

"""Return the concrete index list of a `TensorComponent`."""
indices_of_tensor(e::TensorComponent) = e.indices

"""
    rank_of(e::TensorComponent) -> Int

Number of slots of the expression. Dispatches alongside `rank_of(::Tensor)`.
"""
rank_of(e::TensorComponent) = length(e.indices)

"""
    canonical_slots(e::TensorComponent) -> Vector{Symbol}

Return the canonical slot vbundles for this component.

For a registered [`Tensor`](@ref), these are `tensor.slots` from `@def_tensor`.
For [`KroneckerDelta`](@ref), `[idx_up.vbundle, idx_down.vbundle]` from the
two indices (contravariant then covariant).
"""
function canonical_slots(e::TensorComponent)
    T = e.tensor
    if T isa Tensor
        return T.slots
    elseif T isa KroneckerDelta
        length(e.indices) == 2 ||
            error("canonical_slots: KroneckerDelta component requires 2 indices.")
        return Symbol[e.indices[1].vbundle, e.indices[2].vbundle]
    else
        error("canonical_slots: unsupported tensor type $(typeof(T)).")
    end
end

"""
    variance_matches_canonical(e::TensorComponent) -> Bool

Return `true` if every index in `e` matches the canonical slot variance
declared in `e.tensor.slots`. Useful for diagnostics and for triggering
automatic index raising/lowering in algebraic simplification.
"""
function variance_matches_canonical(e::TensorComponent)
    slots = canonical_slots(e)
    for (idx, slot_vb) in zip(e.indices, slots)
        idx.vbundle == slot_vb || return false
    end
    return true
end


# =========================================
# 5.  Equality and hashing
# =========================================

Base.:(==)(a::TensorComponent, b::TensorComponent) =
    a.tensor === b.tensor && a.indices == b.indices

Base.hash(e::TensorComponent, h::UInt) =
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
    _format_latex(e::TensorComponent) -> String

Produce a LaTeX math-mode string (without surrounding `\$`).

Examples:
    g[-a1, -a2]  → "g_{a_{1} a_{2}}"
    g[a1, a2]    → "g^{a_{1} a_{2}}"
    T[a1, -a2]   → "T^{a_{1}}_{a_{2}}"
"""
function _format_latex(e::TensorComponent)
    function latex_sym(sym::Symbol)
        s = string(sym)
        m = match(r"^([^\d]*)(\d+)$", s)
        m === nothing ? s : "$(m[1])_{$(m[2])}"
    end

    runs = _group_index_runs(e.indices)
    buf  = print_as(e.tensor)
    for (is_cov, syms) in runs
        body = join(latex_sym(s) * " " for s in syms) |> rstrip
        buf *= is_cov ? "_{$body}" : "^{$body}"
    end
    buf
end

"""
    _format_html(e::TensorComponent) -> String
 
Produce an HTML string for Jupyter / Pluto display.
The tensor name is rendered as-is; covariant indices appear in `<sub>` tags
and contravariant indices in `<sup>` tags, with no additional styling.
"""
function _format_html(e::TensorComponent)
    runs = _group_index_runs(e.indices)
    buf  = print_as(e.tensor)
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
    Base.show(io::IO, e::TensorComponent)
 
Plain-text / REPL display. Renders as `name[±idx1, ±idx2, ...]` where
covariant indices are prefixed with `-` and contravariant indices are bare.
 
    g[-a1, -a2]   →  g[-a1, -a2]
    g[a1, a2]     →  g[a1, a2]
    T[a1, -a2]    →  T[a1, -a2]
"""
function Base.show(io::IO, ::MIME"text/plain", e::TensorComponent)
    idx_strs = map(e.indices) do idx
        is_down(idx) ? "-$(idx.symbol)" : "$(idx.symbol)"
    end
    print(io, "$(print_as(e.tensor))[$(join(idx_strs, ", "))]")
end
 
"""
    Base.show(io::IO, ::MIME"text/latex", e::TensorComponent)
 
LaTeX display for IJulia / Jupyter notebooks.
"""
function Base.show(io::IO, ::MIME"text/latex", e::TensorComponent)
    print(io, "\$", _format_latex(e), "\$")
end
 
"""
    Base.show(io::IO, ::MIME"text/html", e::TensorComponent)
 
HTML display for Jupyter / Pluto notebooks.
"""
function Base.show(io::IO, ::MIME"text/html", e::TensorComponent)
    print(io, _format_html(e))
end
 



# =========================================
# Exports
# =========================================

export TensorComponent
export tensor_of, indices_of_tensor, canonical_slots, variance_matches_canonical
# rank_of: already exported from tensors.jl; the TensorComponent method
# is added here via multiple dispatch — no re-export needed.