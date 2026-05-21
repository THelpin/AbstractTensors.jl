# =========================================
# indices.jl — SymbolicTensors.jl
#
# Two concrete index types share the AbstractIndex supertype:
#
#   CoordinateIndex  — labels coordinate-chart slots (∂/dx basis)
#                      registered by @def_manifold (first list) and @add_indices
#   BasisIndex       — labels ordered fibre-basis elements (e/θ or any VBundle)
#                      registered by @def_manifold (second list) and @def_vbundle
#
# Variance is encoded entirely by vbundle:
#     :tangentM   → contravariant (upper)
#     :cotangentM → covariant (lower)
#
# Unary - on an index calls flip; bare bound index is contravariant.
# =========================================


# =========================================
# 1.  AbstractIndex and concrete types
# =========================================

"""
    AbstractIndex

Supertype for index objects used in tensor expressions and basis elements.

Concrete subtypes:
- [`CoordinateIndex`](@ref) — coordinate-chart index (∂/dx)
- [`BasisIndex`](@ref)      — fibre-basis index (e/θ or any VBundle basis)
"""
abstract type AbstractIndex end

"""
    CoordinateIndex

An index symbol for the coordinate chart of a vector bundle.

After `@def_manifold M 4 [a1,a2,a3,a4] [A1,A2,A3,A4]`, each symbol in the
first list is bound as a contravariant `CoordinateIndex`:

     a1          # CoordinateIndex(:a1, :tangentM)   — contravariant
    -a1          # CoordinateIndex(:a1, :cotangentM) — covariant (unary -)

### Fields

- `symbol`  : the index name, e.g. `:a1`
- `vbundle` : the bundle it lives in, e.g. `:tangentM` or `:cotangentM`
"""
struct CoordinateIndex <: AbstractIndex
    symbol::Symbol
    vbundle::Symbol
end

"""
    BasisIndex

An index symbol labelling an element of an ordered fibre basis.

After `@def_manifold M 4 [a1,a2,a3,a4] [A1,A2,A3,A4]`, each symbol in the
second list is bound as a contravariant `BasisIndex`:

     A1          # BasisIndex(:A1, :tangentM)   — contravariant
    -A1          # BasisIndex(:A1, :cotangentM) — covariant (unary -)

Also used by `@def_vbundle` for indices of any custom vector bundle.

### Fields

- `symbol`  : the index name, e.g. `:A1` or `:v1`
- `vbundle` : the home (primal) bundle, e.g. `:tangentM` or `:E`
"""
struct BasisIndex <: AbstractIndex
    symbol::Symbol
    vbundle::Symbol
end


# =========================================
# 2.  Registries
# =========================================

"""
    _COORDINATE_INDICES :: Dict{Symbol, Symbol}

Maps each coordinate index symbol to its home (primal) vbundle.

    _COORDINATE_INDICES[:a1] → :tangentM

Populated by `@def_manifold` (first index list) and `@add_indices`.
"""
const _COORDINATE_INDICES = Dict{Symbol, Symbol}()

"""
    _BASIS_INDICES :: Dict{Symbol, Symbol}

Maps each basis index symbol to its home (primal) vbundle.

    _BASIS_INDICES[:A1] → :tangentM
    _BASIS_INDICES[:v1] → :E

Populated by `@def_manifold` (second index list) and `@def_vbundle`.
"""
const _BASIS_INDICES = Dict{Symbol, Symbol}()


function _register_index!(dict::Dict{Symbol,Symbol}, sym::Symbol, vbundle::Symbol, kind::Symbol)
    if haskey(dict, sym)
        existing = dict[sym]
        existing == vbundle && return
        error(
            "$(kind) index :$sym is already registered to vbundle $existing. " *
            "Cannot re-register to $vbundle. " *
            "Call @undef_manifold or @undef_vbundle on the original bundle first."
        )
    end
    # Also guard against cross-registry collision
    other = kind === :coordinate ? _BASIS_INDICES : _COORDINATE_INDICES
    haskey(other, sym) &&
        error(
            "Index :$sym is already registered as a $(kind === :coordinate ? "basis" : "coordinate") index. " *
            "Symbol names must be unique across both registries."
        )
    dict[sym] = vbundle
end

"""
    register_coordinate_index!(sym::Symbol, vbundle::Symbol)

Register `sym` as a coordinate index belonging to primal vbundle `vbundle`.
"""
register_coordinate_index!(sym::Symbol, vbundle::Symbol) =
    _register_index!(_COORDINATE_INDICES, sym, vbundle, :coordinate)

"""
    register_basis_index!(sym::Symbol, vbundle::Symbol)

Register `sym` as a basis index belonging to primal vbundle `vbundle`.
"""
register_basis_index!(sym::Symbol, vbundle::Symbol) =
    _register_index!(_BASIS_INDICES, sym, vbundle, :basis)

unregister_coordinate_index!(sym::Symbol) = delete!(_COORDINATE_INDICES, sym)
unregister_basis_index!(sym::Symbol)       = delete!(_BASIS_INDICES, sym)

"""
    unregister_index!(sym::Symbol)

Remove `sym` from whichever registry contains it. Silent if not registered.
"""
function unregister_index!(sym::Symbol)
    delete!(_COORDINATE_INDICES, sym)
    delete!(_BASIS_INDICES, sym)
    nothing
end

# ── Registry accessors ────────────────────────────────────────────────────────

is_coordinate_index(sym::Symbol) = haskey(_COORDINATE_INDICES, sym)
is_basis_index(sym::Symbol)       = haskey(_BASIS_INDICES, sym)

"""
    index_kind(sym::Symbol) -> Symbol

Return `:coordinate` or `:basis` for a registered symbol. Errors if unknown.
"""
function index_kind(sym::Symbol)
    is_coordinate_index(sym) && return :coordinate
    is_basis_index(sym)       && return :basis
    error("Index :$sym is not registered.")
end

is_index_registered(sym::Symbol) =
    is_coordinate_index(sym) || is_basis_index(sym)

is_index_registered(t::AbstractIndex) = is_index_registered(t.symbol)

"""
    index_home_vbundle(sym::Symbol) -> Symbol

Return the home (primal) vbundle of `sym`. Errors if not registered.
"""
function index_home_vbundle(sym::Symbol)
    is_coordinate_index(sym) && return _COORDINATE_INDICES[sym]
    is_basis_index(sym)       && return _BASIS_INDICES[sym]
    error("Index :$sym is not registered. Was @def_manifold called?")
end

index_home_vbundle(t::AbstractIndex) = index_home_vbundle(t.symbol)

"""
    basis_indices_for_vbundle(vb::Symbol) -> Vector{BasisIndex}

Return all [`BasisIndex`](@ref) objects registered with home vbundle `vb`.
"""
function basis_indices_for_vbundle(vb::Symbol)
    syms = sort([s for (s, home) in _BASIS_INDICES if home == vb])
    [BasisIndex(s, vb) for s in syms]
end

"""
    up(sym::Symbol) -> AbstractIndex

Return the contravariant (upper) form of registered symbol `sym`.
"""
function up(sym::Symbol)
    home = index_home_vbundle(sym)
    is_coordinate_index(sym) && return CoordinateIndex(sym, home)
    return BasisIndex(sym, home)
end


# =========================================
# 3.  Transformations
# =========================================

function _flip_index(t::AbstractIndex)
    haskey(_VBUNDLES, t.vbundle) ||
        error("VBundle $(t.vbundle) is not registered.")
    dual_vb = _VBUNDLES[t.vbundle].dual
    t isa CoordinateIndex && return CoordinateIndex(t.symbol, dual_vb)
    return BasisIndex(t.symbol, dual_vb)
end

"""
    flip(t::AbstractIndex) -> AbstractIndex

Return a new index with the dual vbundle (toggle variance).
"""
flip(t::AbstractIndex) = _flip_index(t)


# =========================================
# 4.  Unary operators
# =========================================

Base.:-(t::AbstractIndex) = flip(t)
Base.:+(t::AbstractIndex) = t


# =========================================
# 5.  Variance predicates
# =========================================
"""
    is_up(t::AbstractIndex) -> Bool

Return `true` if `t` is contravariant (upper): its `vbundle` is primal
(`isdual == false`), e.g. `:tangentM`.

Also available as `t.is_up` via [`Base.getproperty`](@ref).

# Examples

After `@def_manifold M 4 [a1, a2, a3, a4] [A1, A2, A3, A4]`, `is_up(a1)` is
`true` and `is_up(-a1)` is `false`.

~~~julia
@def_manifold M 4 [a1, a2, a3, a4] [A1, A2, A3, A4]
is_up(a1)   # true
is_up(-a1)  # false
~~~
"""
function is_up(t::AbstractIndex)
    haskey(_VBUNDLES, t.vbundle) || error("VBundle $(t.vbundle) is not registered.")
    !_VBUNDLES[t.vbundle].isdual
end

"""
    is_down(t::AbstractIndex) -> Bool

Return `true` if `t` is covariant (lower): its `vbundle` is dual
(`isdual == true`), e.g. `:cotangentM`.

Also available as `t.is_down` via [`Base.getproperty`](@ref).

# Examples

After `@def_manifold M 4 [a1, a2, a3, a4] [A1, A2, A3, A4]`, `is_down(-a1)` is
`true` and `is_down(a1)` is `false`.

~~~julia
@def_manifold M 4 [a1, a2, a3, a4] [A1, A2, A3, A4]
is_down(-a1)  # true
is_down(a1)   # false
~~~
"""
function is_down(t::AbstractIndex)
    haskey(_VBUNDLES, t.vbundle) || error("VBundle $(t.vbundle) is not registered.")
    _VBUNDLES[t.vbundle].isdual
end

function Base.getproperty(t::AbstractIndex, field::Symbol)
    if field === :is_up
        return is_up(t)
    elseif field === :is_down
        return is_down(t)
    else
        return getfield(t, field)
    end
end

function Base.propertynames(::AbstractIndex, private::Bool=false)
    (:symbol, :vbundle, :is_up, :is_down)
end


# =========================================
# 6.  Predicates
# =========================================

same_symbol(a::AbstractIndex, b::AbstractIndex) = a.symbol == b.symbol

"""
    contractable(a::AbstractIndex, b::AbstractIndex) -> Bool

True if `a` and `b` form a valid Einstein summation pair: same kind,
same symbol, dual vbundles.
"""
function contractable(a::AbstractIndex, b::AbstractIndex)
    typeof(a) === typeof(b) ||
        return false
    same_symbol(a, b) && is_dual_vbundles(a.vbundle, b.vbundle)
end


# =========================================
# 7.  Equality & hashing
# =========================================

Base.:(==)(a::CoordinateIndex, b::CoordinateIndex) =
    a.symbol == b.symbol && a.vbundle == b.vbundle
Base.:(==)(a::BasisIndex, b::BasisIndex) =
    a.symbol == b.symbol && a.vbundle == b.vbundle
Base.:(==)(a::AbstractIndex, b::AbstractIndex) = false

Base.hash(t::CoordinateIndex, h::UInt) = hash((CoordinateIndex, t.symbol, t.vbundle), h)
Base.hash(t::BasisIndex, h::UInt)       = hash((BasisIndex, t.symbol, t.vbundle), h)


# =========================================
# 8.  Display
# =========================================

function Base.show(io::IO, t::AbstractIndex)
    prefix = (haskey(_VBUNDLES, t.vbundle) && _VBUNDLES[t.vbundle].isdual) ? "-" : "+"
    kind   = t isa CoordinateIndex ? "coord" : "basis"
    print(io, "$(prefix)$(t.symbol) ∈ $(t.vbundle) ($(kind))")
end


# =========================================
# 9.  @add_indices macro  (coordinate indices only)
# =========================================

"""
    @add_indices M idx1 idx2 ...

Register extra **coordinate** index symbols to the tangent bundle of manifold
`M` and bind each to a contravariant [`CoordinateIndex`](@ref) in scope.

# Example
~~~julia
@def_manifold M 4 [a1, a2, a3, a4] [A1, A2, A3, A4]
@add_indices M a5 a6
a5   # CoordinateIndex(:a5, :tangentM)
~~~
"""
macro add_indices(manifold_name, idx_syms...)
    isempty(idx_syms) &&
        error("@add_indices: provide at least one index symbol.")
    manifold_name isa Symbol ||
        error("@add_indices: first argument must be a manifold symbol, got $manifold_name.")

    manifold_sym  = QuoteNode(manifold_name)
    tangent_sym   = QuoteNode(Symbol("tangent", manifold_name))
    cotangent_sym = QuoteNode(Symbol("cotangent", manifold_name))
    idx_sym_nodes = [s isa Symbol ? QuoteNode(s) : (s isa QuoteNode && s.value isa Symbol ? s : error("@add_indices: index names must be plain symbols, got $s")) for s in idx_syms]

    assignments = map(idx_syms) do s
        s isa Symbol ||
            error("@add_indices: index names must be plain symbols, got $s.")
        quote
            register_coordinate_index!($(QuoteNode(s)), $(tangent_sym))
            $(esc(s)) = CoordinateIndex($(QuoteNode(s)), $(tangent_sym))
        end
    end

    quote
        haskey(_MANIFOLDS, $(manifold_sym)) ||
            error(
                "@add_indices: manifold $($(manifold_sym)) is not registered. " *
                "Call @def_manifold $($(manifold_sym)) first."
            )
        $(assignments...)
        local _tb_vb  = _VBUNDLES[$(tangent_sym)]
        local _ctb_vb = _VBUNDLES[$(cotangent_sym)]
        local _extra_tb = CoordinateIndex[]
        local _extra_ct = CoordinateIndex[]
        for _s in ($(idx_sym_nodes...),)
            push!(_extra_tb, CoordinateIndex(_s, $(tangent_sym)))
            push!(_extra_ct, CoordinateIndex(_s, $(cotangent_sym)))
        end
        _VBUNDLES[$(tangent_sym)] = VBundle(
            getfield(_tb_vb, :name), getfield(_tb_vb, :manifold), getfield(_tb_vb, :dim),
            getfield(_tb_vb, :isdual), getfield(_tb_vb, :dual),
            vcat(getfield(_tb_vb, :coordinate_indices), _extra_tb),
            getfield(_tb_vb, :basis_indices),
        )
        _VBUNDLES[$(cotangent_sym)] = VBundle(
            getfield(_ctb_vb, :name), getfield(_ctb_vb, :manifold), getfield(_ctb_vb, :dim),
            getfield(_ctb_vb, :isdual), getfield(_ctb_vb, :dual),
            vcat(getfield(_ctb_vb, :coordinate_indices), _extra_ct),
            getfield(_ctb_vb, :basis_indices),
        )
        $(esc(Symbol("tangent", manifold_name)))   = _VBUNDLES[$(tangent_sym)]
        $(esc(Symbol("cotangent", manifold_name))) = _VBUNDLES[$(cotangent_sym)]
        nothing
    end
end


# =========================================
# 10.  Validation helpers
# =========================================

"""
    validate_indices(syms::Vector{Symbol}, vbundle::Symbol)

Check that every symbol in `syms` is a registered **coordinate** index whose
home (primal) vbundle is `vbundle`.

Throws `ErrorException` if any symbol is missing from `_COORDINATE_INDICES` or
its home bundle differs from `vbundle`. Returns `nothing` if all checks pass.

Used internally when validating lists of coordinate symbols (e.g. after
[`@def_manifold`](@ref) or [`@add_indices`](@ref)).

# Examples

~~~julia
@def_manifold M 4 [a1, a2, a3, a4] [A1, A2, A3, A4]
validate_indices([:a1, :a2], :tangentM)   # ok
validate_indices([:a1, :NOT_REG], :tangentM)   # throws — unknown symbol
validate_indices([:a1], :wrongBundle)         # throws — wrong home bundle
~~~
"""
function validate_indices(syms::Vector{Symbol}, vbundle::Symbol)
    for s in syms
        is_coordinate_index(s) ||
            error(
                "Index :$s is not registered as a coordinate index. " *
                "Call @def_manifold or @add_indices first."
            )
        actual = _COORDINATE_INDICES[s]
        actual == vbundle ||
            error(
                "Index :$s has home bundle $actual, " *
                "but expected $vbundle."
            )
    end
end

"""
    validate_contraction(a::AbstractIndex, b::AbstractIndex)

Check that `a` and `b` are a valid Einstein summation pair.

Requires:

1. Same index kind (`CoordinateIndex` with `CoordinateIndex`, or
   `BasisIndex` with `BasisIndex`).
2. Same symbol (`a.symbol == b.symbol`).
3. Dual vbundles ([`is_dual_vbundles`](@ref) on `a.vbundle` and `b.vbundle`).

Throws `ErrorException` if any check fails; returns `nothing` otherwise.

For a non-throwing predicate, see [`contractable`](@ref).

# Examples
~~~julia
@def_manifold M 4 [a1, a2, a3, a4] [A1, A2, A3, A4]
validate_contraction(a1, -a1)   # ok — dual partners, same symbol
validate_contraction(a1, a2)    # throws — different symbols
validate_contraction(a1, a1)    # throws — same vbundle (not dual)
~~~
"""
function validate_contraction(a::AbstractIndex, b::AbstractIndex)
    typeof(a) === typeof(b) ||
        error(
            "Cannot contract $(a.symbol) ($(typeof(a))) with $(b.symbol) ($(typeof(b))): " *
            "indices must be of the same kind (both coordinate or both basis)."
        )
    same_symbol(a, b) ||
        error(
            "Cannot contract $(a.symbol) with $(b.symbol): different symbols."
        )
    is_dual_vbundles(a.vbundle, b.vbundle) ||
        error(
            "Cannot contract $(a.symbol) ($(a.vbundle)) " *
            "with $(b.symbol) ($(b.vbundle)): bundles are not dual partners."
        )
end


# =========================================
# Exports
# =========================================

export AbstractIndex, CoordinateIndex, BasisIndex
export flip, up
export _COORDINATE_INDICES, _BASIS_INDICES
export is_up, is_down
export basis_indices_for_vbundle
export register_coordinate_index!, register_basis_index!, unregister_index!
export @add_indices
