# =========================================
# manifolds.jl — SymbolicTensors.jl
#
# Design principles:
#   - Manifolds and vector bundles are plain struct instances.
#     @def_manifold binds M, tangentM, cotangentM as variables
#     in the caller's scope, all queryable via dot access:
#       M.dim, M.tangent_bundle, tangentM.isdual, etc.
#   - All metadata lives in module-level registries.
#   - Indices are registered via register_coordinate_index! / register_basis_index! from indices.jl.
#   - VBundle.isdual is the single authoritative source for
#     bundle variance (false = tangent, true = cotangent/dual).
#     No naming conventions are relied upon for this.
#   - Coordinate index symbols are bound as contravariant CoordinateIndex;
#     basis index symbols as contravariant BasisIndex.
#
# xTensor analogs:
#   $Manifolds              → _MANIFOLDS
#   ManifoldQ[M]            → is_manifold(M)
#   DimOfManifold[M]        → M.dim
#   TangentBundleOfManifold → M.tangent_bundle
#   IndicesOfVBundle        → vb.coordinate_indices / vb.basis_indices
# =========================================

# Depends on indices.jl being loaded first.
# In SymbolicTensors.jl: include("indices.jl") before include("manifolds.jl")

# =========================================
# 1. Core structs
# =========================================

"""
    Manifold

Struct representing a registered differentiable manifold. Instances are created by
[`@def_manifold`](@ref) and bound to a variable in the caller's scope.

Provides dot access to all metadata:

    M.name              # :M
    M.dim               # 4
    M.tangent_bundle    # :tangentM
    M.cotangent_bundle  # :cotangentM
    M.vbundles          # [:tangentM, :cotangentM]

### Fields

- `name`             : manifold name, e.g. `:M`
- `dim`              : dimension
- `tangent_bundle`   : name of the tangent bundle, e.g. `:tangentM`
- `cotangent_bundle` : name of the cotangent (dual) bundle, e.g. `:cotangentM`
- `vbundles`         : names of vector bundles with manifold as base manifold
"""
struct Manifold
    name::Symbol
    dim::Dim
    tangent_bundle::Symbol
    cotangent_bundle::Symbol
    vbundles::Vector{Symbol}
end

"""
    VBundle

Struct representing a registered vector bundle. Instances are created by [`@def_manifold`](@ref)
for the tangent and cotangent bundles,
and bound to variables in the caller's scope (e.g. `tangentM`, `cotangentM`).

Provides dot access to all metadata:

    tangentM.name      # :tangentM
    tangentM.manifold  # :M
    tangentM.dim       # 4
    tangentM.isdual    # false
    tangentM.dual      # :cotangentM
    tangentM.coordinate_indices  # [CoordinateIndex(:a1, :tangentM), ...]
    tangentM.basis_indices       # [BasisIndex(:A1, :tangentM), ...]
    tangentM.bases     # [Basis(:∂, :tangentM, :coordinate), Basis(:e, :tangentM, :frame)]

### Fields

- `name`     : bundle name, e.g. `:tangentM`
- `manifold` : base manifold name, e.g. `:M`
- `dim`      : fibre dimension
- `isdual`   : false = contravariant (upper) slots, true = covariant (lower) slots.
               Authoritative for index variance via [`is_up`](@ref) / [`is_down`](@ref).
- `dual`     : name of the paired dual bundle, e.g. `:cotangentM` or `:dualE`
- `coordinate_indices` : [`CoordinateIndex`](@ref) for the coordinate chart (∂ / `dx`);
               nonempty on tangent/cotangent bundles from [`@def_manifold`](@ref)
- `basis_indices` : [`BasisIndex`](@ref) for fibre / moving bases (`e` / `θ`);
               populated on tangent/cotangent by [`@def_manifold`](@ref) and on custom
               bundles by [`@def_vbundle`](@ref)
"""
struct VBundle
    name::Symbol
    manifold::Symbol
    dim::Dim
    isdual::Bool
    dual::Symbol
    coordinate_indices::Vector{CoordinateIndex}
    basis_indices::Vector{BasisIndex}
end


# =========================================
# 2. Module-level registries
# =========================================

"""
Registry of all defined manifolds.
Key: manifold name as Symbol (e.g. `:M`).
"""
const _MANIFOLDS = Dict{Symbol, Manifold}()

"""
Registry of all defined vector bundles.
Key: bundle name as Symbol (e.g. `:tangentM`).
"""
const _VBUNDLES = Dict{Symbol, VBundle}()


# =========================================
# 3.  Bundle pairing
# =========================================

"""
    is_dual_vbundles(vb1::Symbol, vb2::Symbol) -> Bool

!!! warning "Internal"
    This function is intended for internal use by the SymbolicTensors.jl
    package. It is not part of the public API and may change without notice.

True if `vb2` is the dual partner of `vb1`, i.e. `_VBUNDLES[vb1].dual == vb2`.

    is_dual_vbundles(:tangentM, :cotangentM)  →  true
    is_dual_vbundles(:E, :dualE)              →  true
    is_dual_vbundles(:E, :cotangentM)         →  false
"""
function is_dual_vbundles(vb1::Symbol, vb2::Symbol)
    haskey(_VBUNDLES, vb1) || return false
    _VBUNDLES[vb1].dual == vb2
end


# =========================================
# 4. Predicates
# =========================================

# Internal predicates (not exported)
is_manifold(x) = x isa Manifold
is_vbundle(x) = x isa VBundle
is_tangent_bundle(v::VBundle) =
    haskey(_MANIFOLDS, v.manifold) && v.name == _MANIFOLDS[v.manifold].tangent_bundle
function is_tangent_bundle(vb::Symbol)
    haskey(_VBUNDLES, vb) || return false
    m = _MANIFOLDS[_VBUNDLES[vb].manifold]
    vb == m.tangent_bundle
end

is_tangent_bundle(::Any)      = false

is_cotangent_bundle(v::VBundle) =
    haskey(_MANIFOLDS, v.manifold) && v.name == _MANIFOLDS[v.manifold].cotangent_bundle
function is_cotangent_bundle(vb::Symbol)
    haskey(_VBUNDLES, vb) || return false
    m = _MANIFOLDS[_VBUNDLES[vb].manifold]
    vb == m.cotangent_bundle
end

is_cotangent_bundle(::Any)      = false


# =========================================
# 5. Helper: parse index symbols at macro expansion time
# =========================================

function _macro_index_symbols(indices_expr)::Vector{Symbol}
    Meta.isexpr(indices_expr, :vect) ||
        error("@def_manifold: indices must be a vector literal like [a1, a2, a3, a4], got $indices_expr")
    out = Symbol[]
    for a in indices_expr.args
        if a isa Symbol
            push!(out, a)
        elseif a isa QuoteNode && a.value isa Symbol
            push!(out, a.value)
        else
            error("@def_manifold: each index must be a plain symbol, got $a")
        end
    end
    return out
end


# =========================================
# 6. @def_manifold macro
# =========================================


# ── shared helpers for inline frame registration ──────────────────────────────

# Generate an inline code block that registers a COORDINATE frame in _BASES
# (key: (vbundle, :coordinate)) and binds the basis variables in the CALLER's
# scope (via esc at the calling macro's level).
#
# Called at MACRO EXPANSION TIME from within @def_manifold.
# Using a plain function (not a nested macro) ensures that esc() is relative to
# the outermost macro call site, correctly binding variables in the user's scope.
#
# primal_q    : QuoteNode for the primal vbundle (e.g. QuoteNode(:tangentM))
# dual_q      : QuoteNode for the dual   vbundle (e.g. QuoteNode(:cotangentM))
# basis_sym   : Symbol to bind for the primal coordinate frame (e.g. :∂)
# cobasis_sym : Symbol to bind for the dual   coordinate frame (e.g. :dx)
function _gen_coord_frame_registration_expr(
    primal_q    :: QuoteNode,
    dual_q      :: QuoteNode,
    basis_sym   :: Symbol,
    cobasis_sym :: Symbol,
)
    bq          = QuoteNode(basis_sym)
    cq          = QuoteNode(cobasis_sym)
    primal_key  = QuoteNode((primal_q.value, :coordinate))
    dual_key    = QuoteNode((dual_q.value,   :coordinate))
    quote
        _BASES[$(primal_key)] = Basis($(bq), $(primal_q), :coordinate)
        _BASES[$(dual_key)]   = Basis($(cq), $(dual_q),   :coordinate)
        $(esc(basis_sym))     = _BASES[$(primal_key)]
        $(esc(cobasis_sym))   = _BASES[$(dual_key)]
        println(
            "Defined vector bundle $($(primal_q)) with coordinate basis $($(bq)) " *
            "and its dual $($(dual_q)) with coordinate basis $($(cq))"
        )
        nothing
    end
end

# Generate an inline code block that registers a MOVING frame in _BASES
# (key: (vbundle, :frame)), creates FrameBundle structs, and binds all four
# variables in the CALLER's scope via esc.
#
# frame_name   : Symbol for the primal frame bundle (e.g. :frameM)
# coframe_name : Symbol for the dual   frame bundle (e.g. :coframeM)
# primal_q     : QuoteNode for the primal vbundle (e.g. QuoteNode(:tangentM))
# dual_q       : QuoteNode for the dual   vbundle (e.g. QuoteNode(:cotangentM))
# basis_sym    : Symbol to bind for the primal moving frame (e.g. :e)
# cobasis_sym  : Symbol to bind for the dual   moving frame (e.g. :θ)
# manifold_q   : QuoteNode for the manifold name (for the print message)
function _gen_moving_frame_registration_expr(
    frame_name   :: Symbol,
    coframe_name :: Symbol,
    primal_q     :: QuoteNode,
    dual_q       :: QuoteNode,
    basis_sym    :: Symbol,
    cobasis_sym  :: Symbol,
    manifold_q   :: QuoteNode,
)
    bq           = QuoteNode(basis_sym)
    cq           = QuoteNode(cobasis_sym)
    fn_q         = QuoteNode(frame_name)
    cfn_q        = QuoteNode(coframe_name)
    primal_key   = QuoteNode((primal_q.value, :frame))
    dual_key     = QuoteNode((dual_q.value,   :frame))
    quote
        _BASES[$(primal_key)] = Basis($(bq), $(primal_q), :frame)
        _BASES[$(dual_key)]   = Basis($(cq), $(dual_q),   :frame)

        _FRAME_BUNDLES[$(fn_q)]  = FrameBundle($(fn_q),  $(primal_q), $(cfn_q), _BASES[$(primal_key)])
        _FRAME_BUNDLES[$(cfn_q)] = FrameBundle($(cfn_q), $(dual_q),   $(fn_q),  _BASES[$(dual_key)])

        $(esc(frame_name))    = _FRAME_BUNDLES[$(fn_q)]
        $(esc(coframe_name))  = _FRAME_BUNDLES[$(cfn_q)]
        $(esc(basis_sym))     = _BASES[$(primal_key)]
        $(esc(cobasis_sym))   = _BASES[$(dual_key)]

        println(
            "Defined frame bundle $($(fn_q)) (moving frame $($(bq))) " *
            "and coframe bundle $($(cfn_q)) (moving coframe $($(cq))) " *
            "over $($(manifold_q))"
        )
        nothing
    end
end

# ── kwargs parser for @def_manifold ──────────────────────────────────────────

# Parse natural_frame=, natural_coframe=, moving_frame=, moving_coframe= kwargs.
# Returns (nat_frame::Symbol, nat_coframe::Symbol, mov_frame::Symbol, mov_coframe::Symbol)
# with defaults :∂, :dx, :e, :θ.
function _parse_manifold_kwargs(kwargs)
    nat_frame    = :∂    # coordinate frame for tangent bundle
    nat_coframe  = :dx   # coordinate frame for cotangent bundle
    mov_frame    = :e    # moving frame for tangent bundle
    mov_coframe  = :θ    # moving frame for cotangent bundle

    for kw in kwargs
        Meta.isexpr(kw, :(=), 2) ||
            error("@def_manifold: expected keyword=value argument, got: $kw")
        k, v = kw.args

        sym_val = if v isa Symbol
            v
        elseif v isa QuoteNode && v.value isa Symbol
            v.value
        else
            nothing
        end

        if k === :natural_frame
            sym_val !== nothing ||
                error("@def_manifold: natural_frame must be a symbol, got $v")
            nat_frame = sym_val
        elseif k === :natural_coframe
            sym_val !== nothing ||
                error("@def_manifold: natural_coframe must be a symbol, got $v")
            nat_coframe = sym_val
        elseif k === :moving_frame
            sym_val !== nothing ||
                error("@def_manifold: moving_frame must be a symbol, got $v")
            mov_frame = sym_val
        elseif k === :moving_coframe
            sym_val !== nothing ||
                error("@def_manifold: moving_coframe must be a symbol, got $v")
            mov_coframe = sym_val
        else
            error(
                "@def_manifold: unknown keyword :$k. " *
                "Supported: natural_frame, natural_coframe, moving_frame, moving_coframe."
            )
        end
    end

    # Validate uniqueness across all four names
    all_names = (nat_frame, nat_coframe, mov_frame, mov_coframe)
    length(unique(all_names)) == 4 ||
        error(
            "@def_manifold: all four frame names must be distinct; got $all_names"
        )

    return nat_frame, nat_coframe, mov_frame, mov_coframe
end

"""
    @def_manifold name dim coord_indices basis_indices [kwargs...]

Define a new manifold and automatically create its tangent and cotangent
bundles, coordinate frames, and moving frame bundles.

Both index lists are **required**. Each list should have at least 4 symbols
(a warning is issued if fewer).

Bind in the caller's scope:
- `name`            → a [`Manifold`](@ref) instance
- `tangent<name>`   → a [`VBundle`](@ref) instance (`isdual = false`)
- `cotangent<name>` → a [`VBundle`](@ref) instance (`isdual = true`)
- `frame<name>`, `coframe<name>`, moving basis symbols (default `e`, `θ`)

Coordinate indices (first list) → [`CoordinateIndex`](@ref):

    a1          # CoordinateIndex(:a1, :tangentM)   — contravariant
    -a1         # CoordinateIndex(:a1, :cotangentM) — covariant

Basis indices (second list) → [`BasisIndex`](@ref):

    A1          # BasisIndex(:A1, :tangentM)   — contravariant
    -A1         # BasisIndex(:A1, :cotangentM) — covariant

#### Examples
```julia
@def_manifold M 4 [a1, a2, a3, a4] [A1, A2, A3, A4]
@def_manifold M d [b1, b2, b3, b4] [B1, B2, B3, B4]   # parametric dimension
```
"""
macro def_manifold(name, dim, coord_indices, basis_indices, kwargs...)
    name isa Symbol ||
        error("@def_manifold: first argument must be a symbol, got $name")

    # Parse kwargs — returns (nat_frame, nat_coframe, mov_frame, mov_coframe)
    nat_frame, nat_coframe, mov_frame, mov_coframe = _parse_manifold_kwargs(kwargs)

    dim_expr = if dim isa Integer
        dim
    elseif dim isa Symbol
        QuoteNode(dim)
    else
        esc(dim)
    end
    tangent_name     = Symbol("tangent",   name)
    cotangent_name   = Symbol("cotangent", name)
    frame_name       = Symbol("frame",   name)
    coframe_name     = Symbol("coframe", name)
    name_symbol      = QuoteNode(name)
    tangent_symbol   = QuoteNode(tangent_name)
    cotangent_symbol = QuoteNode(cotangent_name)
    coord_symbols    = _macro_index_symbols(coord_indices)
    basis_symbols    = _macro_index_symbols(basis_indices)

    quote
        # ── Guard: clean up stale registry entries if redefining ─────────
        if haskey(_MANIFOLDS, $(name_symbol))
            @warn "Manifold $($(name_symbol)) is already defined. Redefining."
            local _old_tb  = getfield(_MANIFOLDS[$(name_symbol)], :tangent_bundle)
            local _old_ctb = getfield(_MANIFOLDS[$(name_symbol)], :cotangent_bundle)
            if haskey(_VBUNDLES, _old_tb)
                local _old_vb = _VBUNDLES[_old_tb]
                for _old_idx in getfield(_old_vb, :coordinate_indices)
                    unregister_index!(getfield(_old_idx, :symbol))
                end
                for _old_idx in getfield(_old_vb, :basis_indices)
                    unregister_index!(getfield(_old_idx, :symbol))
                end
                delete!(_VBUNDLES, _old_tb)
                delete!(_VBUNDLES, _old_ctb)
            end
            # Clean up all frame types from _BASES
            for _ftype in (:coordinate, :frame)
                delete!(_BASES, (_old_tb,  _ftype))
                delete!(_BASES, (_old_ctb, _ftype))
            end
            # Clean up frame bundles
            local _old_fb  = Symbol("frame",   $(name_symbol))
            local _old_cfb = Symbol("coframe", $(name_symbol))
            delete!(_FRAME_BUNDLES, _old_fb)
            delete!(_FRAME_BUNDLES, _old_cfb)
            delete!(_MANIFOLDS, $(name_symbol))
        end

        # ── Runtime locals ───────────────────────────────────────────────
        local _dim::Dim = $(dim_expr)
        local _coord_indices  = $(coord_symbols)
        local _basis_indices  = $(basis_symbols)

        _dim isa Int && (_dim > 0 || error("@def_manifold: dimension must be positive, got $_dim"))

        if length(_coord_indices) < 4
            @warn "Manifold $($(name_symbol)): fewer coordinate indices ($(length(_coord_indices))) " *
                  "than 4. Add more with @add_indices later."
        end
        if length(_basis_indices) < 4
            @warn "Manifold $($(name_symbol)): fewer basis indices ($(length(_basis_indices))) " *
                  "than 4."
        end

        # ── Step 1: Register manifold stub ────────────────────────────────
        println("Defined manifold $($(name_symbol)) of dimension $(_dim)")

        # ── Step 2: Register coordinate indices ─────────────────────────────
        for _idx in _coord_indices
            register_coordinate_index!(_idx, $(tangent_symbol))
        end

        # ── Step 3: Register basis indices ────────────────────────────────
        for _bidx in _basis_indices
            register_basis_index!(_bidx, $(tangent_symbol))
        end

        # ── Step 4: Build index vectors for VBundle ───────────────────────
        local _t_coord = [CoordinateIndex(s, $(tangent_symbol))   for s in _coord_indices]
        local _c_coord = [CoordinateIndex(s, $(cotangent_symbol)) for s in _coord_indices]
        local _t_basis = [BasisIndex(s, $(tangent_symbol))   for s in _basis_indices]
        local _c_basis = [BasisIndex(s, $(cotangent_symbol)) for s in _basis_indices]

        # ── Step 5: Register bundles ─────────────────────────────────────
        _VBUNDLES[$(tangent_symbol)] = VBundle(
            $(tangent_symbol), $(name_symbol), _dim, false,
            $(cotangent_symbol), _t_coord, _t_basis
        )
        _VBUNDLES[$(cotangent_symbol)] = VBundle(
            $(cotangent_symbol), $(name_symbol), _dim, true,
            $(tangent_symbol), _c_coord, _c_basis
        )

        # ── Step 6: Register manifold ─────────────────────────────────────
        _MANIFOLDS[$(name_symbol)] = Manifold(
            $(name_symbol), _dim,
            $(tangent_symbol), $(cotangent_symbol),
            [$(tangent_symbol), $(cotangent_symbol)]
        )

        # ── Step 7: Bind Manifold and VBundle instances in caller's scope ─
        $(esc(name))           = _MANIFOLDS[$(name_symbol)]
        $(esc(tangent_name))   = _VBUNDLES[$(tangent_symbol)]
        $(esc(cotangent_name)) = _VBUNDLES[$(cotangent_symbol)]

        # ── Step 8: Bind CoordinateIndex variables in caller's scope ─────
        $([ :($(esc(s)) = CoordinateIndex($(QuoteNode(s)), $(tangent_symbol))) for s in coord_symbols ]...)

        # ── Step 9: Bind BasisIndex variables in caller's scope ───────────
        $([ :($(esc(s)) = BasisIndex($(QuoteNode(s)), $(tangent_symbol))) for s in basis_symbols ]...)

        # ── Step 10: Register coordinate frame (inline) ───────────────────
        $(_gen_coord_frame_registration_expr(tangent_symbol, cotangent_symbol, nat_frame, nat_coframe))

        # ── Step 11: Register moving frame (inline) ───────────────────────
        $(_gen_moving_frame_registration_expr(
            frame_name, coframe_name,
            tangent_symbol, cotangent_symbol,
            mov_frame, mov_coframe,
            name_symbol,
        ))

        nothing
    end
end


# =========================================
# 7.  @undef_manifold macro
# =========================================

"""
    @undef_manifold name

Remove a manifold and all its associated bundles and index registrations
from the module-level registries.

After this call:
- `_MANIFOLDS[name]`  no longer exists
- `_VBUNDLES[tangentM]` and `_VBUNDLES[cotangentM]` no longer exist
- every coordinate and basis index symbol registered to the tangent bundle is unregistered
  from `_COORDINATE_INDICES` and `_BASIS_INDICES`

## Stale variable warning

Julia module-level bindings cannot be deleted at runtime. The variable
`name` in the caller's scope will still exist and still hold the old
`Manifold` struct after this call. Attempting to access any field on
that stale reference will raise an immediate warning:

```julia
@def_manifold M 4 [a1, a2, a3, a4] [A1, A2, A3, A4]
@undef_manifold M

M.dim   # → Warning: Manifold :M has been undefined. Variable still holds a stale reference.
```

To fully clear the name from your session, restart the Julia kernel.
"""
macro undef_manifold(name)
    name isa Symbol ||
        error("@undef_manifold: argument must be a symbol, got $name")

    name_sym = QuoteNode(name)

    quote
        haskey(_MANIFOLDS, $(name_sym)) ||
            error(
                "@undef_manifold: manifold $($(name_sym)) is not registered. " *
                "Call @def_manifold $($(name_sym)) first, or check keys(_MANIFOLDS)."
            )

        local _m_data  = _MANIFOLDS[$(name_sym)]

        # Capture names BEFORE deletion — getfield bypasses the stale guard.
        local _tb_name  = getfield(_m_data, :tangent_bundle)
        local _ctb_name = getfield(_m_data, :cotangent_bundle)

        if haskey(_VBUNDLES, _tb_name)
            local _tb_vb = _VBUNDLES[_tb_name]
            for _t_idx in getfield(_tb_vb, :coordinate_indices)
                unregister_index!(getfield(_t_idx, :symbol))
            end
            for _t_idx in getfield(_tb_vb, :basis_indices)
                unregister_index!(getfield(_t_idx, :symbol))
            end
            delete!(_VBUNDLES, _tb_name)
            delete!(_VBUNDLES, _ctb_name)
        end

        # Clean up all frame types from _BASES
        for _ftype in (:coordinate, :frame)
            delete!(_BASES, (_tb_name,  _ftype))
            delete!(_BASES, (_ctb_name, _ftype))
        end

        # Clean up frame bundles
        local _fb_name  = Symbol("frame",   $(name_sym))
        local _cfb_name = Symbol("coframe", $(name_sym))
        delete!(_FRAME_BUNDLES, _fb_name)
        delete!(_FRAME_BUNDLES, _cfb_name)

        delete!(_MANIFOLDS, $(name_sym))

        @warn "Manifold $($(name_sym)) has been undefined. " *
              "The variable $($(name_sym)) still holds a stale reference," *
              " accessing it will error. Restart the kernel to fully clear the binding."
        println("Undefined tangent bundle:   $_tb_name")
        println("Undefined cotangent bundle: $_ctb_name")
        nothing
    end
end

# =========================================
# 8.  Stale-reference guards
# =========================================

"""
    Base.getproperty(m::Manifold, field::Symbol)

Field access for `Manifold` instances.

Before returning the requested field, checks that `m` is still registered
in `_MANIFOLDS`. If `@undef_manifold` has been called, the variable in the
caller's scope may still hold the old struct (Julia cannot delete bindings
at runtime). This guard turns that silent stale reference into an immediate,
descriptive warning.
"""
function Base.getproperty(m::Manifold, field::Symbol)
    if !haskey(_MANIFOLDS, getfield(m, :name))
        @warn "Manifold :$(getfield(m, :name)) has been undefined. " *
              "Variable still holds a stale reference."
        return nothing
    end
    getfield(m, field)
end

"""
    Base.getproperty(v::VBundle, field::Symbol)

Field access for `VBundle` instances.

Same stale-reference guard as for `Manifold`: checks that `v` is still
registered in `_VBUNDLES` before returning the requested field.

In addition, exposes the virtual property `:bases` which returns all
[`Basis`](@ref) objects registered for this bundle in `_BASES` (defined in
`frames.jl`, populated by [`@def_manifold`](@ref) and
[`@def_frame_bundle`](@ref)). Returns an empty vector if no frames
have been registered yet.

    tangentM.bases
    # → [Basis(:∂, :tangentM, :coordinate), Basis(:e, :tangentM, :frame)]

    cotangentM.bases
    # → [Basis(:dx, :cotangentM, :coordinate), Basis(:θ, :cotangentM, :frame)]

The `_BASES` lookup is resolved at call time, so `frames.jl` need not
be loaded before `manifolds.jl` — no forward-reference problem arises.
"""
function Base.getproperty(v::VBundle, field::Symbol)
    if field === :bases
        return bases_for_vbundle(getfield(v, :name))
    end
    if !haskey(_VBUNDLES, getfield(v, :name))
        @warn "VBundle :$(getfield(v, :name)) has been undefined. " *
              "Variable still holds a stale reference."
        return nothing
    end
    getfield(v, field)
end

"""
    Base.propertynames(v::VBundle, private::Bool=false)

Return the property names available on a `VBundle`, including the
virtual `:bases` property backed by `_BASES`.
"""
function Base.propertynames(::VBundle, private::Bool=false)
    (:name, :manifold, :dim, :isdual, :dual, :coordinate_indices, :basis_indices, :bases)
end

# =========================================
# 9. Utility / introspection
# =========================================

"""
    list_manifolds() -> Vector{Symbol}

Return the names of all currently registered manifolds.
"""
list_manifolds() = collect(keys(_MANIFOLDS))

# =========================================
# 10. show methods
# =========================================

function Base.show(io::IO, M::Manifold)
    print(io, "Manifold($(M.name), dim=$(M.dim), " *
              "TBundle=$(M.tangent_bundle), CBundle=$(M.cotangent_bundle))")
end

function Base.show(io::IO, ::MIME"text/html", M::Manifold)
    print(io, """
    <div style="border:1px solid #ddd;padding:10px;border-radius:5px;background:#f4faff;">
        <h4 style="margin-top:0;">Manifold: <span style="color:#0d6efd;">$(M.name)</span></h4>
        <table style="width:100%;border-collapse:collapse;">
            <tr><td style="font-weight:bold;width:150px;text-align:left;">Dimension</td><td>$(M.dim)</td></tr>
            <tr><td style="font-weight:bold;text-align:left;">Tangent Bundle</td><td><code>$(M.tangent_bundle)</code></td></tr>
            <tr><td style="font-weight:bold;text-align:left;">Cotangent Bundle</td><td><code>$(M.cotangent_bundle)</code></td></tr>
            <tr><td style="font-weight:bold;text-align:left;">All VBundles</td>
                <td>$(join(map(x -> "<code>$x</code>", M.vbundles), ", "))</td></tr>
        </table>
    </div>
    """)
end


# =========================================
# Exports
# =========================================

export Manifold, VBundle
export _MANIFOLDS, _VBUNDLES
export @def_manifold, @undef_manifold