# =========================================
# indices.jl — AbstractTensor.jl
#
# — IndexSymbol
#   A named index symbol bound to a specific manifold's tangent bundle.
#   Created in the caller's scope by @def_manifold and @add_indices.
#   Provides property-style access to registry metadata and can be
#   passed directly to up() and down().
#
# — TensorIndex
#   A flat (symbol, vbundle) struct representing an index in a tensor
#   expression. The vbundle encodes the variance completely:
#     :TangentM   → contravariant (upper) index
#     :CoTangentM → covariant (lower) index
#
#   up / down are input-syntax helpers only. When the user writes T{a, -b},
#   the parser calls up(a) or down(b) — where a and b are IndexSymbol objects
#   or plain Symbols — which resolves the correct vbundle from the registry
#   and constructs a TensorIndex. After construction, no position field is
#   needed or stored.
#
# xTensor analogs:
#   abstract index `a`        →  IndexSymbol(:a) bound to name `a` in scope
#   upper slot  `a` in T[a]   →  up(a)    = TensorIndex(:a, :TangentM)
#   lower slot `-a` in T[-a]  →  down(a)  = TensorIndex(:a, :CoTangentM)
# =========================================


# =========================================
# 0.  IndexSymbol
# =========================================
 
"""
    IndexSymbol
 
A named index symbol bound to a specific manifold's tangent bundle,
created by [`@def_manifold`](@ref) and [`@add_indices`](@ref) in the
caller's scope.
 
Provides property-style access to registry metadata:
 
    a1.symbol    # :a1
    a1.vbundle   # :TangentM
 
Pass directly to [`up`](@ref) and [`down`](@ref), or use unary `-` / `+`:
 
    up(a1)    # TensorIndex(:a1, :TangentM)
    down(a1)  # TensorIndex(:a1, :CoTangentM)
    -a1       # same as down(a1)  — for `T[-a1, ...]` sugar
    +a1       # same as up(a1)
 
Fields
------
- `symbol` : the index name as a `Symbol`, e.g. `:a1`
"""
struct IndexSymbol
    symbol::Symbol
end
 
function Base.getproperty(i::IndexSymbol, field::Symbol)
    if field == :vbundle
        return index_home_vbundle(i.symbol)
    else
        return getfield(i, field)   # fallback for actual struct fields (:symbol)
    end
end
 
function Base.propertynames(::IndexSymbol, private::Bool=false)
    (:symbol, :vbundle)
end
 
function Base.show(io::IO, i::IndexSymbol)
    if haskey(_INDICES, i.symbol)
        vb = index_home_vbundle(i.symbol)
        print(io, "$(i.symbol) ∈ $vb")
    else
        print(io, "$(i.symbol) (unregistered)")
    end
end


"""
    Base.:-(i::IndexSymbol) -> TensorIndex

Unary minus on an [`IndexSymbol`](@ref): returns `down(i)` (covariant / lower index).
Enables bracket sugar such as `F[-a1, -a2]` when indexing a [`Tensor`](@ref).
"""
Base.:-(i::IndexSymbol) = down(i)

"""
    Base.:+(i::IndexSymbol) -> TensorIndex

Unary plus on an [`IndexSymbol`](@ref): returns `up(i)` (contravariant / upper index).
"""
Base.:+(i::IndexSymbol) = up(i)

# =========================================
# 1.  TensorIndex
# =========================================

"""
    TensorIndex

An index symbol placed in a specific vector bundle, fully encoding its
variance through the bundle it lives in:

- vbundle = `:TangentM`   → contravariant (upper) index
- vbundle = `:CoTangentM` → covariant (lower) index

[`up`](@ref) and [`down`](@ref) are the standard constructors. They accept
either a plain `Symbol` or an [`IndexSymbol`](@ref) and resolve the correct
vbundle from the registry at construction time.

Construction
------------
    up(:a1)    →  TensorIndex(:a1, :TangentM)      # contravariant, Symbol form
    down(:a1)  →  TensorIndex(:a1, :CoTangentM)    # covariant,     Symbol form
    up(a1)     →  TensorIndex(:a1, :TangentM)      # contravariant, IndexSymbol form
    down(a1)   →  TensorIndex(:a1, :CoTangentM)    # covariant,     IndexSymbol form

Direct construction is valid when both fields are known:

    TensorIndex(:a1, :TangentM)

Fields
------
- `symbol`  : the index name, e.g. `:a1`
- `vbundle` : the bundle it lives in, e.g. `:TangentM` or `:CoTangentM`

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

    _INDICES[:a1]  →  :TangentM

Every index is registered to its tangent bundle only. The cotangent bundle
is reached via `dual_vbundle` (defined in manifolds.jl). This is the single
source of truth: `up` reads from here to get the tangent bundle, `down`
calls `dual_vbundle` on that result to get the cotangent bundle.

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
is_index_registered(sym::Symbol)    = haskey(_INDICES, sym)
is_index_registered(i::IndexSymbol) = is_index_registered(i.symbol)

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
index_home_vbundle(i::IndexSymbol) = index_home_vbundle(i.symbol)


# =========================================
# 3.  Constructors  (up / down as syntax layer)
# =========================================

"""
    up(sym::Symbol)    -> TensorIndex
    up(i::IndexSymbol) -> TensorIndex

Construct a contravariant (upper) index.
Looks up the home tangent bundle from the registry.

    up(:a1)  →  TensorIndex(:a1, :TangentM)
    up(a1)   →  TensorIndex(:a1, :TangentM)   # a1 an IndexSymbol
"""
function up(sym::Symbol)
    vb = index_home_vbundle(sym)
    TensorIndex(sym, vb)
end
up(i::IndexSymbol) = up(i.symbol)

"""
    down(sym::Symbol)    -> TensorIndex
    down(i::IndexSymbol) -> TensorIndex

Construct a covariant (lower) index.
Looks up the home tangent bundle, then takes its dual (cotangent bundle).

    down(:a1)  →  TensorIndex(:a1, :CoTangentM)
    down(a1)   →  TensorIndex(:a1, :CoTangentM)   # a1 an IndexSymbol

Requires `dual_vbundle` to be defined (manifolds.jl, loaded after this file).
"""
function down(sym::Symbol)
    vb = index_home_vbundle(sym)
    TensorIndex(sym, dual_vbundle(vb))
end
down(i::IndexSymbol) = down(i.symbol)


# =========================================
# 4.  Variance predicates
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
# 5.  Predicates
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
    dual_vbundles(vb1::Symbol, vb2::Symbol) -> Bool

!!! warning "Internal"
    This function is intended for internal use by the AbstractTensors.jl
    package. It is not part of the public API and may change without notice.

True if `vb1` and `vb2` are the tangent/cotangent pair of the same manifold.
Reads from `_VBUNDLES` (defined in manifolds.jl).
"""
function dual_vbundles(vb1::Symbol, vb2::Symbol)
    (haskey(_VBUNDLES, vb1) && haskey(_VBUNDLES, vb2)) || return false
    r1, r2 = _VBUNDLES[vb1], _VBUNDLES[vb2]
    r1.manifold == r2.manifold && r1.isdual != r2.isdual
end

"""
    contractable(a::TensorIndex, b::TensorIndex) -> Bool

!!! warning "Internal"
    This function is intended for internal use by the AbstractTensors.jl
    package. It is not part of the public API and may change without notice.

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

Requires `dual_vbundle` (manifolds.jl).
"""
flip(t::TensorIndex) = TensorIndex(t.symbol, dual_vbundle(t.vbundle))


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
    prefix = (haskey(_VBUNDLES, t.vbundle) && _VBUNDLES[t.vbundle].isdual) ? "-" : " "
    print(io, "$(prefix)$(t.symbol) ∈ $(t.vbundle)")
end


# =========================================
# 9.  @add_indices macro
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
bind each to an [`IndexSymbol`](@ref) in the caller's scope.

`@def_manifold` already registers its index list, so `@add_indices` is
useful for introducing additional indices after manifold definition.

Requires that `M` has already been defined with `@def_manifold`.

# Example
```julia
@def_manifold M 4 [a1, a2, a3, a4]
@add_indices M a5 a6
a5.vbundle              # :TangentM
up(a5)                  # TensorIndex(:a5, :TangentM)
down(a6)                # TensorIndex(:a6, :CoTangentM)
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
            $(esc(s)) = IndexSymbol($(QuoteNode(s)))
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
# 10.  Validation helpers  (used by tensors.jl)
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

export IndexSymbol
export TensorIndex
export up, down, flip
export _INDICES
export is_up, is_down
export @add_indices