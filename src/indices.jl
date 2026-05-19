# =========================================
# indices.jl — AbstractTensor.jl
#
# — TensorIndex
#   A flat (symbol, vbundle) struct representing an index in a tensor
#   expression. The vbundle encodes the variance completely:
#     :tangentM   → contravariant (upper) index
#     :cotangentM → covariant (lower) index
#
#   When the user writes T[-a1, a2] or F[-a1, -a2], unary - on a TensorIndex
#   calls flip; a bare bound TensorIndex is already contravariant.
#   Variance is encoded only by vbundle — no separate position field.
#
# Design change (from previous version):
#   IndexSymbol has been removed. @def_manifold and @def_vbundle now bind
#   TensorIndex(:sym, :tangentX) directly in the caller's scope.
#   Unary - and + are defined on TensorIndex (flip / identity).
#   This eliminates the Vector{Any} heterogeneity and reduces the
#   conceptual surface area for users.
#
# xTensor analogs:
#   abstract index `a`        →  TensorIndex(:a, :tangentM) bound to name `a`
#   upper slot  `a` in T[a]   →  a itself  (already contravariant)
#   lower slot `-a` in T[-a]  →  -a  = flip(a) = TensorIndex(:a, :cotangentM)
# =========================================


# =========================================
# 1.  TensorIndex
# =========================================

"""
    TensorIndex

An index symbol associated to a specific vector bundle, fully encoding its
variance through the bundle it lives in:

- vbundle = `:tangentM`   → contravariant (upper) index
- vbundle = `:cotangentM` → covariant (lower) index

After `@def_manifold M 4 [a1, a2, a3, a4]`, each index symbol is bound in
the caller's scope as a contravariant `TensorIndex`:

     a1          # TensorIndex(:a1, :tangentM)   — contravariant
    -a1          # TensorIndex(:a1, :cotangentM) — covariant (unary -)
    +a1          # TensorIndex(:a1, :tangentM)   — contravariant (unary +, identity)
    flip(a1)     # Change vbundle of index to its dual

Bracket indexing uses bound `TensorIndex` values only: `F[-a1, -a2]`.

Direct construction when both fields are known:

    TensorIndex(:a1, :tangentM)

### Fields

- `symbol`  : the index name, e.g. `:a1`
- `vbundle` : the bundle it lives in, e.g. `:tangentM` or `:cotangentM`

Dot access (via `getproperty`) also exposes:

    idx.is_up    # upper / contravariant
    idx.is_down  # lower / covariant
"""
struct TensorIndex
    symbol::Symbol
    vbundle::Symbol
end


# =========================================
# 2.  Registry
# =========================================

"""
    _INDICES :: Dict{Symbol, Symbol}

Maps each registered index symbol to the name of its *home* (tangent) bundle.

    _INDICES[:a1]  →  :tangentM

Every index is registered to its tangent bundle only. The cotangent bundle
is reached via the `dual` field on [`VBundle`](@ref) (manifolds.jl) and [`flip`](@ref).

Populated by `register_index!` (called from `@def_manifold` and `@add_indices`).
Cleared entry-by-entry by `unregister_index!` (called from `@undef_manifold`).

Do not mutate directly — use the API below.
"""
const _INDICES = Dict{Symbol, Symbol}()   # symbol → home vbundle (tangent)

"""
    register_index!(sym::Symbol, vbundle::Symbol)

!!! warning "Internal"
    This function is intended for internal use by the AbstractTensors.jl
    package. It is not part of the public API and may change without notice.

Register `sym` as belonging to `vbundle` (always the tangent bundle).

- Idempotent: re-registering to the *same* vbundle is a no-op.
- Errors if `sym` is already registered to a *different* vbundle.
  An index belongs to exactly one home bundle.

Called by `@def_manifold` and `@add_indices`.
"""
function register_index!(sym::Symbol, vbundle::Symbol)
    if haskey(_INDICES, sym)
        existing = _INDICES[sym]
        existing == vbundle && return            # idempotent
        error(
            "Index :$sym is already registered to vbundle $existing. " *
            "Cannot re-register to $vbundle. " *
            "Each index belongs to exactly one home bundle. " *
            "Call @undef_manifold on the original manifold first."
        )
    end
    _INDICES[sym] = vbundle
end

"""
    unregister_index!(sym::Symbol)

!!! warning "Internal"
    This function is intended for internal use by the AbstractTensors.jl
    package. It is not part of the public API and may change without notice.

Remove `sym` from the registry. Silent if `sym` was not registered.
Called by `@undef_manifold`.
"""
unregister_index!(sym::Symbol) = delete!(_INDICES, sym)

# ── Registry accessors ────────────────────────────────────────────────────────

"""
    is_index_registered(sym::Symbol) -> Bool

!!! warning "Internal"
    This function is intended for internal use by the AbstractTensors.jl
    package. It is not part of the public API and may change without notice.

True if `sym` is in [`_INDICES`](@ref).
"""
is_index_registered(sym::Symbol)       = haskey(_INDICES, sym)
is_index_registered(t::TensorIndex)    = is_index_registered(t.symbol)

"""
    index_home_vbundle(sym::Symbol) -> Symbol

!!! warning "Internal"
    This function is intended for internal use by the AbstractTensors.jl
    package. It is not part of the public API and may change without notice.

Return the home (tangent) vbundle of `sym`. Errors if not registered.
"""
function index_home_vbundle(sym::Symbol)
    haskey(_INDICES, sym) ||
        error("Index :$sym is not registered. Was @def_manifold called?")
    _INDICES[sym]
end
index_home_vbundle(t::TensorIndex) = index_home_vbundle(t.symbol)


# =========================================
# 3.  Transformations
# =========================================

"""
    flip(t::TensorIndex) -> TensorIndex

Return a new `TensorIndex` with the dual vbundle.
Contravariant → covariant and vice versa.

Reads `dual` from [`_VBUNDLES`](@ref) (manifolds.jl).
"""
function flip(t::TensorIndex)
    haskey(_VBUNDLES, t.vbundle) ||
        error("VBundle $(t.vbundle) is not registered.")
    TensorIndex(t.symbol, _VBUNDLES[t.vbundle].dual)
end


# =========================================
# 4.  Unary operators on TensorIndex
#     Enables: -a1  (covariant sugar)  and  +a1  (identity)
#     Since @def_manifold binds a1 = TensorIndex(:a1, :tangentM),
#     [a1, -a2] now produces Vector{TensorIndex} uniformly.
# =========================================

"""
    Base.:-(t::TensorIndex) -> TensorIndex

Unary minus on a [`TensorIndex`](@ref): returns `flip(t)` (toggles variance).
Enables bracket sugar such as `F[-a1, -a2]` when indexing a [`Tensor`](@ref).

    -a1   →  TensorIndex(:a1, :cotangentM)   # if a1 is contravariant
    -a1   →  TensorIndex(:a1, :tangentM)     # if a1 is covariant  (double flip)
"""
Base.:-(t::TensorIndex) = flip(t)

"""
    Base.:+(t::TensorIndex) -> TensorIndex

Unary plus on a [`TensorIndex`](@ref): identity, returns `t` unchanged.
"""
Base.:+(t::TensorIndex) = t


# =========================================
# 5.  Variance predicates
# =========================================

"""
    is_up(t::TensorIndex) -> Bool

True if `t` is an upper (contravariant) index — lives in the tangent bundle
(`isdual = false`). Equivalent to `t.is_up`.
"""
function is_up(t::TensorIndex)
    haskey(_VBUNDLES, t.vbundle) || error("VBundle $(t.vbundle) is not registered.")
    !_VBUNDLES[t.vbundle].isdual
end

"""
    is_down(t::TensorIndex) -> Bool

True if `t` is a lower (covariant) index — lives in the cotangent (dual) bundle
(`isdual = true`). Equivalent to `t.is_down`.
"""
function is_down(t::TensorIndex)
    haskey(_VBUNDLES, t.vbundle) || error("VBundle $(t.vbundle) is not registered.")
    _VBUNDLES[t.vbundle].isdual
end

function Base.getproperty(t::TensorIndex, field::Symbol)
    if field === :is_up
        return is_up(t)
    elseif field === :is_down
        return is_down(t)
    else
        return getfield(t, field)
    end
end

function Base.propertynames(::TensorIndex, private::Bool=false)
    (:symbol, :vbundle, :is_up, :is_down)
end


# =========================================
# 6.  Predicates
# =========================================

"""
    same_symbol(a::TensorIndex, b::TensorIndex) -> Bool

!!! warning "Internal"
    This function is intended for internal use by the AbstractTensors.jl
    package. It is not part of the public API and may change without notice.

True if both indices carry the same symbol, regardless of bundle.
Used to detect repeated indices (Einstein summation candidates).
"""
same_symbol(a::TensorIndex, b::TensorIndex) = a.symbol == b.symbol

"""
    contractable(a::TensorIndex, b::TensorIndex) -> Bool

!!! warning "Internal"
    This function is intended for internal use by the AbstractTensors.jl
    package. It is not part of the public API and may change without notice.

True if `a` and `b` form a valid Einstein summation pair:
same symbol, and their vbundles are registered dual partners
(see [`is_dual_vbundles`](@ref)).
"""
function contractable(a::TensorIndex, b::TensorIndex)
    same_symbol(a, b) && is_dual_vbundles(a.vbundle, b.vbundle)
end


# =========================================
# 8.  Equality & hashing
# =========================================

Base.:(==)(a::TensorIndex, b::TensorIndex) =
    a.symbol == b.symbol && a.vbundle == b.vbundle

Base.hash(t::TensorIndex, h::UInt) = hash((t.symbol, t.vbundle), h)


# =========================================
# 9.  Display
# =========================================

function Base.show(io::IO, t::TensorIndex)
    prefix = (haskey(_VBUNDLES, t.vbundle) && _VBUNDLES[t.vbundle].isdual) ? "-" : " "
    print(io, "$(prefix)$(t.symbol) ∈ $(t.vbundle)")
end


# =========================================
# 10.  @add_indices macro
# =========================================

# Note: there is no @remove_indices macro.
# Removing an individual index after definition is unsafe — any tensor
# already defined using that index would be left in an inconsistent state
# with no way to detect it at runtime. Index removal is only performed
# as part of a full @undef_manifold teardown, which coordinates cleanup
# across _INDICES, _VBUNDLES, and _MANIFOLDS atomically.

"""
    @add_indices M idx1 idx2 ...

Register extra index symbols to the tangent bundle of manifold `M` and
bind each to a contravariant [`TensorIndex`](@ref) in the caller's scope.

`@def_manifold` already registers its index list, so `@add_indices` is
useful for introducing additional indices after manifold definition.

Requires that `M` has already been defined with `@def_manifold`.

# Example
```julia
@def_manifold M 4 [a1, a2, a3, a4]
@add_indices M a5 a6
a5                  # TensorIndex(:a5, :tangentM)
-a6                 # TensorIndex(:a6, :cotangentM)
```
"""
macro add_indices(manifold_name, idx_syms...)
    isempty(idx_syms) &&
        error("@add_indices: provide at least one index symbol.")
    manifold_name isa Symbol ||
        error("@add_indices: first argument must be a manifold symbol, got $manifold_name.")

    manifold_sym = QuoteNode(manifold_name)
    tangent_sym  = QuoteNode(Symbol("tangent", manifold_name))

    assignments = map(idx_syms) do s
        s isa Symbol ||
            error("@add_indices: index names must be plain symbols, got $s.")
        quote
            register_index!($(QuoteNode(s)), $(tangent_sym))
            $(esc(s)) = TensorIndex($(QuoteNode(s)), $(tangent_sym))
        end
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
# 11.  Validation helpers  (used by tensors.jl)
# =========================================

"""
    validate_indices(syms::Vector{Symbol}, vbundle::Symbol)

!!! warning "Internal"
    This function is intended for internal use by the AbstractTensors.jl
    package. It is not part of the public API and may change without notice.

Assert that every symbol in `syms` is registered and its home bundle
matches `vbundle`. Throws a descriptive error on the first violation.

Called by `@def_tensor` at definition time.
"""
function validate_indices(syms::Vector{Symbol}, vbundle::Symbol)
    for s in syms
        haskey(_INDICES, s) ||
            error(
                "Index :$s is not registered. " *
                "Call @def_manifold or @add_indices first."
            )
        actual = _INDICES[s]
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

!!! warning "Internal"
    This function is intended for internal use by the AbstractTensors.jl
    package. It is not part of the public API and may change without notice.

Throw a descriptive error if `a` and `b` cannot be contracted.
"""
function validate_contraction(a::TensorIndex, b::TensorIndex)
    same_symbol(a, b) ||
        error(
            "Cannot contract $(a.symbol) with $(b.symbol): different symbols. " *
            "Contraction requires the same index in dual bundles."
        )
    is_dual_vbundles(a.vbundle, b.vbundle) ||
        error(
            "Cannot contract $(a.symbol) ($(a.vbundle)) " *
            "with $(b.symbol) ($(b.vbundle)): " *
            "bundles are not dual partners."
        )
end


# =========================================
# Exports
# =========================================

export TensorIndex
export flip
export _INDICES
export is_up, is_down
export @add_indices