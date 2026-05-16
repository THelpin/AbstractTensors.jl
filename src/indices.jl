# =========================================
# indices.jl — AbstractTensor.jl
#
# — TensorIndex
#   A flat (symbol, vbundle) struct representing an index in a tensor
#   expression. The vbundle encodes the variance completely:
#     :TangentM   → contravariant (upper) index
#     :CoTangentM → covariant (lower) index
#
#   Up / Down are input-syntax helpers only. When the user writes T{a, -b},
#   the parser calls up(:a) or down(:b), which resolves the correct vbundle
#   from the registry (tangent or cotangent) and constructs a TensorIndex.
#   After construction, no position field is needed or stored.
#
# xTensor analogs:
#   abstract index `a`        →  registered symbol :a in _IDX_REGISTRY
#   upper slot  `a` in T[a]   →  up(:a)   = TensorIndex(:a, :TangentM)
#   lower slot `-a` in T[-a]  →  down(:a) = TensorIndex(:a, :CoTangentM)
# =========================================


# =========================================
# 1.  TensorIndex
# =========================================

"""
    TensorIndex

An index symbol placed in a specific vector bundle, fully encoding its
variance through the bundle it lives in:

- vbundle = `:TangentM`   → contravariant (upper) index
- vbundle = `:CoTangentM` → covariant (lower) index

`Up` / `Down` are input-syntax conveniences (see `up`, `down` below).
They are resolved to the correct bundle at construction time and are
not stored in the struct.

Construction
------------
    up(:μ)    →  TensorIndex(:μ, :TangentM)      # contravariant
    down(:μ)  →  TensorIndex(:μ, :CoTangentM)    # covariant

Direct construction is valid when both fields are known:

    TensorIndex(:μ, :TangentM)

Fields
------
- `symbol`  : the index name, e.g. `:μ`
- `vbundle` : the bundle it lives in, e.g. `:TangentM` or `:CoTangentM`
"""
struct TensorIndex
    symbol::Symbol
    vbundle::Symbol
end


# =========================================
# 2.  Registry
# =========================================

"""
    _IDX_REGISTRY :: Dict{Symbol, Symbol}

Maps each registered index symbol to the name of its *home* (tangent) bundle.

    _IDX_REGISTRY[:μ]  →  :TangentM

Every index is registered to its tangent bundle only. The cotangent bundle
is reached via `dual_bundle` (defined in manifolds.jl). This is the single
source of truth: `up` reads from here to get the tangent bundle, `down`
calls `dual_bundle` on that result to get the cotangent bundle.

Populated by `register_index!` (called from `@def_manifold` and `@indices`).
Cleared entry-by-entry by `unregister_index!` (called from `@undef_manifold`).

Do not mutate directly — use the API below.
"""
const _IDX_REGISTRY = Dict{Symbol, Symbol}()   # symbol → home vbundle (tangent)

"""
    register_index!(sym::Symbol, vbundle::Symbol)

Register `sym` as belonging to `vbundle` (always the tangent bundle).

- Idempotent: re-registering to the *same* vbundle is a no-op.
- Errors if `sym` is already registered to a *different* vbundle.
  An index belongs to exactly one home bundle.

Called by `@def_manifold` and `@indices`. Rarely needed in user code.
"""
function register_index!(sym::Symbol, vbundle::Symbol)
    if haskey(_IDX_REGISTRY, sym)
        existing = _IDX_REGISTRY[sym]
        existing == vbundle && return            # idempotent
        error(
            "Index :$sym is already registered to vbundle $existing. " *
            "Cannot re-register to $vbundle. " *
            "Each index belongs to exactly one home bundle. " *
            "Call @undef_manifold on the original manifold first."
        )
    end
    _IDX_REGISTRY[sym] = vbundle
end

"""
    unregister_index!(sym::Symbol)

Remove `sym` from the registry. Silent if `sym` was not registered.
Called by `@undef_manifold`.
"""
unregister_index!(sym::Symbol) = delete!(_IDX_REGISTRY, sym)

# ── Registry accessors ────────────────────────────────────────────────────────

"""
    index_registered(sym::Symbol) -> Bool

True if `sym` is in the registry.
"""
index_registered(sym::Symbol) = haskey(_IDX_REGISTRY, sym)

"""
    index_home_vbundle(sym::Symbol) -> Symbol

Return the home (tangent) vbundle of `sym`. Errors if not registered.
"""
function index_home_vbundle(sym::Symbol)
    haskey(_IDX_REGISTRY, sym) ||
        error("Index :$sym is not registered. Was @def_manifold called?")
    _IDX_REGISTRY[sym]
end


# =========================================
# 3.  Constructors  (up / down as syntax layer)
# =========================================

"""
    up(sym::Symbol) -> TensorIndex

Construct a contravariant (upper) index for `sym`.
Looks up the home tangent bundle from the registry.

    up(:μ)  →  TensorIndex(:μ, :TangentM)
"""
function up(sym::Symbol)
    vb = index_home_vbundle(sym)   # :TangentM
    TensorIndex(sym, vb)
end

"""
    down(sym::Symbol) -> TensorIndex

Construct a covariant (lower) index for `sym`.
Looks up the home tangent bundle, then takes its dual (cotangent bundle).

    down(:μ)  →  TensorIndex(:μ, :CoTangentM)

Requires `dual_bundle` to be defined (manifolds.jl, loaded after this file).
"""
function down(sym::Symbol)
    vb = index_home_vbundle(sym)   # :TangentM
    TensorIndex(sym, dual_bundle(vb))   # :CoTangentM
end


# =========================================
# 4.  Accessors
# =========================================

"""Symbol name of the index, e.g. `:μ`."""
symbol_of(t::TensorIndex)  = t.symbol

"""Vbundle name, e.g. `:TangentM` or `:CoTangentM`."""
vbundle_of(t::TensorIndex) = t.vbundle

"""True if `t` is contravariant — lives in the tangent bundle (isdual = false)."""
function is_contravariant(t::TensorIndex)
    haskey(_VBUNDLES, t.vbundle) || error("Vbundle $(t.vbundle) is not registered.")
    !_VBUNDLES[t.vbundle].isdual
end

"""True if `t` is covariant — lives in the cotangent (dual) bundle (isdual = true)."""
function is_covariant(t::TensorIndex)
    haskey(_VBUNDLES, t.vbundle) || error("Vbundle $(t.vbundle) is not registered.")
    _VBUNDLES[t.vbundle].isdual
end

"""Alias for [`is_contravariant`](@ref). True if `t` is an upper (contravariant) index."""
is_up(t::TensorIndex)   = is_contravariant(t)

"""Alias for [`is_covariant`](@ref). True if `t` is a lower (covariant) index."""
is_down(t::TensorIndex) = is_covariant(t)


# =========================================
# 5.  Predicates
# =========================================

"""
    same_symbol(a::TensorIndex, b::TensorIndex) -> Bool

True if both indices carry the same symbol, regardless of bundle.
Used to detect repeated indices (Einstein summation candidates).
"""
same_symbol(a::TensorIndex, b::TensorIndex) = a.symbol == b.symbol

"""
    dual_vbundles(vb1::Symbol, vb2::Symbol) -> Bool

True if `vb1` and `vb2` are the tangent/cotangent pair of the same manifold.
Reads from `_VBUNDLES` and `_MANIFOLDS` (defined in manifolds.jl).
"""
function dual_vbundles(vb1::Symbol, vb2::Symbol)
    (haskey(_VBUNDLES, vb1) && haskey(_VBUNDLES, vb2)) || return false
    r1, r2 = _VBUNDLES[vb1], _VBUNDLES[vb2]
    r1.manifold == r2.manifold && r1.isdual != r2.isdual
end

"""
    contractable(a::TensorIndex, b::TensorIndex) -> Bool

True if `a` and `b` form a valid Einstein summation pair:
same symbol, and their vbundles are the tangent/cotangent dual pair
of the same manifold.
"""
function contractable(a::TensorIndex, b::TensorIndex)
    same_symbol(a, b) && dual_vbundles(a.vbundle, b.vbundle)
end


# =========================================
# 6.  Transformations
# =========================================

"""
    flip(t::TensorIndex) -> TensorIndex

Return a new `TensorIndex` with the dual vbundle.
Contravariant → covariant and vice versa.

Requires `dual_bundle` (manifolds.jl).
"""
flip(t::TensorIndex) = TensorIndex(t.symbol, dual_bundle(t.vbundle))


# =========================================
# 7.  Equality & hashing
# =========================================

Base.:(==)(a::TensorIndex, b::TensorIndex) =
    a.symbol == b.symbol && a.vbundle == b.vbundle

Base.hash(t::TensorIndex, h::UInt) = hash((t.symbol, t.vbundle), h)


# =========================================
# 8.  Display
# =========================================

function Base.show(io::IO, t::TensorIndex)
    # Determine prefix from isdual if registry is available, else fallback
    prefix = (haskey(_VBUNDLES, t.vbundle) && _VBUNDLES[t.vbundle].isdual) ? "-" : " "
    print(io, "$(prefix)$(t.symbol) ∈ $(t.vbundle)")
end


# =========================================
# 9.  @addIndices macro
# =========================================

"""
    @add_indices M idx1 idx2 ...

Register extra index symbols to the tangent bundle of manifold `M`,
and bind each to its symbol in the caller's scope.

`@def_manifold` already registers its index list, so `@indices` is mainly
useful for introducing additional indices after manifold definition.

Requires that `M` has already been defined with `@def_manifold`.

# Example
```julia
@def_manifold M 4 [μ, ν, ρ, σ]
@add_indices M α β        # register two extra indices on TangentM
index_home_vbundle(:α)  # :TangentM
up(:α)                  # TensorIndex(:α, :TangentM)
down(:α)                # TensorIndex(:α, :CoTangentM)
```
"""
macro add_indices(manifold_name, idx_syms...)
    isempty(idx_syms) &&
        error("@add_indices: provide at least one index symbol.")
    manifold_name isa Symbol ||
        error("@add_indices: first argument must be a manifold symbol, got $manifold_name.")

    manifold_sym = QuoteNode(manifold_name)
    tangent_sym  = QuoteNode(Symbol("Tangent", manifold_name))

    assignments = map(idx_syms) do s
        s isa Symbol ||
            error("@add_indices: index names must be plain symbols, got $s.")
        :( register_index!($(QuoteNode(s)), $(tangent_sym)) )
    end

    quote
        haskey(_MANIFOLDS, $(manifold_sym)) ||
            error(
                "@add_indices: manifold $($(manifold_sym)) is not registered. " *
                "Call @def_manifold $($(manifold_sym)) first."
            )
        $(assignments...)
        nothing
    end
end


# =========================================
# 10.  Validation helpers  (used by tensors.jl)
# =========================================

"""
    validate_indices(syms::Vector{Symbol}, vbundle::Symbol)

Assert that every symbol in `syms` is registered and its home bundle
matches `vbundle`. Throws a descriptive error on the first violation.

Called by `@defTensor` at definition time.
"""
function validate_indices(syms::Vector{Symbol}, vbundle::Symbol)
    for s in syms
        haskey(_IDX_REGISTRY, s) ||
            error(
                "Index :$s is not registered. " *
                "Call @def_manifold or @add_indices first."
            )
        actual = _IDX_REGISTRY[s]
        actual == vbundle ||
            error(
                "Index :$s has home bundle $actual, " *
                "but expected $vbundle. " *
                "All indices of a tensor must share the same home bundle."
            )
    end
end

"""
    validate_contraction(a::TensorIndex, b::TensorIndex)

Throw a descriptive error if `a` and `b` cannot be contracted.
"""
function validate_contraction(a::TensorIndex, b::TensorIndex)
    same_symbol(a, b) ||
        error(
            "Cannot contract $(a.symbol) with $(b.symbol): different symbols. " *
            "Contraction requires the same index in dual bundles."
        )
    dual_vbundles(a.vbundle, b.vbundle) ||
        error(
            "Cannot contract $(a.symbol) ($(a.vbundle)) " *
            "with $(b.symbol) ($(b.vbundle)): " *
            "bundles are not the tangent/cotangent dual pair of the same manifold."
        )
end


# =========================================
# Exports
# =========================================

export TensorIndex
export up, down
export register_index!, unregister_index!
export index_registered, index_home_vbundle
export _IDX_REGISTRY
export symbol_of, vbundle_of
export is_contravariant, is_covariant, is_up, is_down
export same_symbol, dual_vbundles, contractable
export flip
export validate_indices, validate_contraction
export @add_indices