# =========================================
# manifolds.jl — SymbolicTensors.jl
#
# Design principles:
#   - Manifolds and vector bundles are plain struct instances.
#     @def_manifold binds M, tangentM, cotangentM as variables
#     in the caller's scope, all queryable via dot access:
#       M.dim, M.tangent_bundle, tangentM.isdual, etc.
#   - All metadata lives in module-level registries.
#   - Indices are registered via register_coordinate_index! /
#     register_frame_index! from indices.jl.
#   - VBundle.isdual is the single authoritative source for
#     bundle variance (false = tangent, true = cotangent/dual).
#     No naming conventions are relied upon for this.
#   - Coordinate index symbols are bound as contravariant CoordinateIndex;
#     frame index symbols as contravariant FrameIndex.
#
# xTensor analogs:
#   $Manifolds              → _MANIFOLDS
#   ManifoldQ[M]            → is_manifold(M)
#   DimOfManifold[M]        → M.dim
#   TangentBundleOfManifold → M.tangent_bundle
#   IndicesOfVBundle        → vb.coordinate_indices / vb.frame_indices
# =========================================

# Depends on indices.jl being loaded first.

# =========================================
# 1. Core structs
# =========================================

"""
    Manifold

Struct representing a registered differentiable manifold. Instances are
created by [`@def_manifold`](@ref) and bound to a variable in the caller's
scope.

Provides dot access to all metadata:

    M.name              # :M
    M.dim               # 4
    M.tangent_bundle    # :tangentM
    M.cotangent_bundle  # :cotangentM
    M.vbundles          # [:tangentM, :cotangentM]

### Fields

- `name`             : manifold name, e.g. `:M`
- `dim`              : dimension (concrete `Int` or symbolic `Symbol`)
- `tangent_bundle`   : name of the tangent bundle, e.g. `:tangentM`
- `cotangent_bundle` : name of the cotangent (dual) bundle, e.g. `:cotangentM`
- `vbundles`         : names of all vector bundles over this manifold
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

Struct representing a registered vector bundle. Instances are created by
[`@def_manifold`](@ref) for the tangent and cotangent bundles, and by
[`@def_vbundle`](@ref) for custom bundles. Bound to variables in the
caller's scope (e.g. `tangentM`, `cotangentM`).

Provides dot access to all metadata:

    tangentM.name             # :tangentM
    tangentM.manifold         # :M
    tangentM.dim              # 4
    tangentM.isdual           # false
    tangentM.dual             # :cotangentM
    tangentM.coordinate_indices  # [CoordinateIndex(:a1, :tangentM), ...]
    tangentM.frame_indices       # [FrameIndex(:A1, :tangentM), ...]
    tangentM.bases            # [Basis(:∂, :tangentM, :coordinate),
                              #  Basis(:e, :tangentM, :frame)]

### Fields

- `name`               : bundle name, e.g. `:tangentM`
- `manifold`           : base manifold name, e.g. `:M`
- `dim`                : fibre dimension
- `isdual`             : `false` = primal (contravariant), `true` = dual
                         (covariant). Authoritative for variance via
                         [`is_up`](@ref) / [`is_down`](@ref).
- `dual`               : name of the paired dual bundle
- `coordinate_indices` : [`CoordinateIndex`](@ref) objects for the coordinate
                         chart; nonempty for tangent/cotangent bundles
- `frame_indices`      : [`FrameIndex`](@ref) objects for the fibre frame;
                         populated by [`@def_manifold`](@ref) and
                         [`@def_vbundle`](@ref)
"""
struct VBundle
    name::Symbol
    manifold::Symbol
    dim::Dim
    isdual::Bool
    dual::Symbol
    coordinate_indices::Vector{CoordinateIndex}
    frame_indices::Vector{FrameIndex}
end


# =========================================
# 2. Module-level registries
# =========================================

"""
    _MANIFOLDS :: Dict{Symbol, Manifold}

Registry of all defined manifolds. Key: manifold name as `Symbol`.
"""
const _MANIFOLDS = Dict{Symbol, Manifold}()

"""
    _VBUNDLES :: Dict{Symbol, VBundle}

Registry of all defined vector bundles. Key: bundle name as `Symbol`.
"""
const _VBUNDLES = Dict{Symbol, VBundle}()


# =========================================
# 3.  Bundle pairing
# =========================================

"""
    is_dual_vbundles(vb1::Symbol, vb2::Symbol) -> Bool

!!! warning "Internal"

True if `vb2` is the dual partner of `vb1`.

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

is_manifold(x) = x isa Manifold
is_vbundle(x)  = x isa VBundle

is_tangent_bundle(v::VBundle) =
    haskey(_MANIFOLDS, v.manifold) && v.name == _MANIFOLDS[v.manifold].tangent_bundle
function is_tangent_bundle(vb::Symbol)
    haskey(_VBUNDLES, vb) || return false
    _MANIFOLDS[_VBUNDLES[vb].manifold].tangent_bundle == vb
end
is_tangent_bundle(::Any) = false

is_cotangent_bundle(v::VBundle) =
    haskey(_MANIFOLDS, v.manifold) && v.name == _MANIFOLDS[v.manifold].cotangent_bundle
function is_cotangent_bundle(vb::Symbol)
    haskey(_VBUNDLES, vb) || return false
    _MANIFOLDS[_VBUNDLES[vb].manifold].cotangent_bundle == vb
end
is_cotangent_bundle(::Any) = false


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

# ── helper: register coordinate frame in _BASES and bind variables ──────────
function _gen_coord_frame_registration_expr(
    primal_q    :: QuoteNode,
    dual_q      :: QuoteNode,
    basis_sym   :: Symbol,
    cobasis_sym :: Symbol,
)
    bq         = QuoteNode(basis_sym)
    cq         = QuoteNode(cobasis_sym)
    primal_key = QuoteNode((primal_q.value, :coordinate))
    dual_key   = QuoteNode((dual_q.value,   :coordinate))
    quote
        _BASES[$(primal_key)] = Basis($(bq), $(primal_q), :coordinate)
        _BASES[$(dual_key)]   = Basis($(cq), $(dual_q),   :coordinate)
        $(esc(basis_sym))     = _BASES[$(primal_key)]
        $(esc(cobasis_sym))   = _BASES[$(dual_key)]
        println(
            "Defined coordinate frame $($(bq)) on $($(primal_q)) " *
            "and coordinate coframe $($(cq)) on $($(dual_q))"
        )
        nothing
    end
end

# ── helper: register moving frame in _BASES, create FrameBundles, bind vars ─
function _gen_moving_frame_registration_expr(
    frame_name   :: Symbol,
    coframe_name :: Symbol,
    primal_q     :: QuoteNode,
    dual_q       :: QuoteNode,
    basis_sym    :: Symbol,
    cobasis_sym  :: Symbol,
    manifold_q   :: QuoteNode,
)
    bq         = QuoteNode(basis_sym)
    cq         = QuoteNode(cobasis_sym)
    fn_q       = QuoteNode(frame_name)
    cfn_q      = QuoteNode(coframe_name)
    primal_key = QuoteNode((primal_q.value, :frame))
    dual_key   = QuoteNode((dual_q.value,   :frame))
    quote
        _BASES[$(primal_key)] = Basis($(bq), $(primal_q), :frame)
        _BASES[$(dual_key)]   = Basis($(cq), $(dual_q),   :frame)

        _FRAME_BUNDLES[$(fn_q)]  = FrameBundle($(fn_q),  $(primal_q), $(cfn_q), _BASES[$(primal_key)])
        _FRAME_BUNDLES[$(cfn_q)] = FrameBundle($(cfn_q), $(dual_q),   $(fn_q),  _BASES[$(dual_key)])

        $(esc(frame_name))   = _FRAME_BUNDLES[$(fn_q)]
        $(esc(coframe_name)) = _FRAME_BUNDLES[$(cfn_q)]
        $(esc(basis_sym))    = _BASES[$(primal_key)]
        $(esc(cobasis_sym))  = _BASES[$(dual_key)]

        println(
            "Defined frame bundle $($(fn_q)) (moving frame $($(bq))) " *
            "and coframe bundle $($(cfn_q)) (moving coframe $($(cq))) " *
            "over $($(manifold_q))"
        )
        nothing
    end
end

# ── kwargs parser ────────────────────────────────────────────────────────────
function _parse_manifold_kwargs(kwargs)
    coord_frame    = :∂    # coordinate frame for tangent bundle
    coord_coframe  = :dx   # coordinate frame for cotangent bundle
    moving_frame   = :e    # moving frame for tangent bundle
    moving_coframe = :θ    # moving frame for cotangent bundle

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

        if k === :coord_frame
            sym_val !== nothing ||
                error("@def_manifold: coord_frame must be a symbol, got $v")
            coord_frame = sym_val
        elseif k === :coord_coframe
            sym_val !== nothing ||
                error("@def_manifold: coord_coframe must be a symbol, got $v")
            coord_coframe = sym_val
        elseif k === :moving_frame
            sym_val !== nothing ||
                error("@def_manifold: moving_frame must be a symbol, got $v")
            moving_frame = sym_val
        elseif k === :moving_coframe
            sym_val !== nothing ||
                error("@def_manifold: moving_coframe must be a symbol, got $v")
            moving_coframe = sym_val
        else
            error(
                "@def_manifold: unknown keyword :$k. " *
                "Supported: coord_frame, coord_coframe, moving_frame, moving_coframe."
            )
        end
    end

    all_names = (coord_frame, coord_coframe, moving_frame, moving_coframe)
    length(unique(all_names)) == 4 ||
        error("@def_manifold: all four frame names must be distinct; got $all_names")

    return coord_frame, coord_coframe, moving_frame, moving_coframe
end

"""
    @def_manifold <name> dim coord_indices frame_indices [kwargs...]

Define a new manifold and automatically create its tangent and cotangent
bundles, coordinate frames, and moving frame bundles.

Both index lists are **required**. Each list should have at least 4 symbols
(a warning is issued if fewer).

`dim` accepts a concrete positive integer or a bare symbol (e.g. `d`) for
parametric / general-dimension calculations where the dimension is not fixed
at definition time.

### Bindings in the caller's scope

- `<name>`            → [`Manifold`](@ref) instance
- `tangent<name>`     → [`VBundle`](@ref) (`isdual = false`)
- `cotangent<name>`   → [`VBundle`](@ref) (`isdual = true`)
- `frame<name>`       → [`FrameBundle`](@ref) (moving frame bundle)
- `coframe<name>`     → [`FrameBundle`](@ref) (moving coframe bundle)
- coordinate frame / coframe symbols (default `∂`, `dx`)
- moving frame / coframe symbols (default `e`, `θ`)
- each symbol in `coord_indices` → [`CoordinateIndex`](@ref) (contravariant)
- each symbol in `frame_indices` → [`FrameIndex`](@ref) (contravariant)

### Index variance

Coordinate indices:

    a1          # CoordinateIndex(:a1, :tangentM)   — contravariant
    -a1         # CoordinateIndex(:a1, :cotangentM) — covariant

Frame indices:

    A1          # FrameIndex(:A1, :tangentM)   — contravariant
    -A1         # FrameIndex(:A1, :cotangentM) — covariant

### Keyword arguments

| Keyword          | Default | Description                                           |
|:-----------------|:--------|:------------------------------------------------------|
| `coord_frame`    | `:∂`    | Basis name for the coordinate frame on the tangent bundle   |
| `coord_coframe`  | `:dx`   | Basis name for the coordinate frame on the cotangent bundle |
| `moving_frame`   | `:e`    | Basis name for the moving frame on the tangent bundle       |
| `moving_coframe` | `:θ`    | Basis name for the moving frame on the cotangent bundle     |

All four names must be distinct.

### Examples

~~~julia
# Minimal — concrete dimension, default frame names
@def_manifold M 4 [a1, a2, a3, a4] [A1, A2, A3, A4]
# Binds: M, tangentM, cotangentM, frameM, coframeM,
#        ∂, dx (coordinate frames), e, θ (moving frames),
#        a1, a2, a3, a4 (CoordinateIndex), A1, A2, A3, A4 (FrameIndex)

# Parametric dimension
@def_manifold M d [a1, a2, a3, a4] [A1, A2, A3, A4]

# Custom frame names — all four must be distinct
@def_manifold M 4 [a1, a2, a3, a4] [A1, A2, A3, A4]   \\
    coord_frame=:e_coord   coord_coframe=:θ_coord       \\
    moving_frame=:e_mov    moving_coframe=:θ_mov
~~~
"""
macro def_manifold(name, dim, coord_indices, frame_indices, kwargs...)
    name isa Symbol ||
        error("@def_manifold: first argument must be a symbol, got $name")

    coord_frame, coord_coframe, moving_frame, moving_coframe =
        _parse_manifold_kwargs(kwargs)

    dim_expr = if dim isa Integer
        dim
    elseif dim isa Symbol
        QuoteNode(dim)
    else
        esc(dim)
    end

    tangent_name     = Symbol("tangent",   name)
    cotangent_name   = Symbol("cotangent", name)
    frame_bundle_name   = Symbol("frame",   name)
    coframe_bundle_name = Symbol("coframe", name)

    name_symbol      = QuoteNode(name)
    tangent_symbol   = QuoteNode(tangent_name)
    cotangent_symbol = QuoteNode(cotangent_name)
    coord_symbols    = _macro_index_symbols(coord_indices)
    frame_symbols    = _macro_index_symbols(frame_indices)

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
                for _old_idx in getfield(_old_vb, :frame_indices)
                    unregister_index!(getfield(_old_idx, :symbol))
                end
                delete!(_VBUNDLES, _old_tb)
                delete!(_VBUNDLES, _old_ctb)
            end
            for _ftype in (:coordinate, :frame)
                delete!(_BASES, (_old_tb,  _ftype))
                delete!(_BASES, (_old_ctb, _ftype))
            end
            local _old_fb  = Symbol("frame",   $(name_symbol))
            local _old_cfb = Symbol("coframe", $(name_symbol))
            delete!(_FRAME_BUNDLES, _old_fb)
            delete!(_FRAME_BUNDLES, _old_cfb)
            delete!(_MANIFOLDS, $(name_symbol))
        end

        # ── Runtime locals ───────────────────────────────────────────────
        local _dim::Dim          = $(dim_expr)
        local _coord_syms        = $(coord_symbols)
        local _frame_syms        = $(frame_symbols)

        _dim isa Int && (_dim > 0 ||
            error("@def_manifold: dimension must be positive, got $_dim"))

        if length(_coord_syms) < 4
            @warn "Manifold $($(name_symbol)): fewer coordinate indices " *
                  "($(length(_coord_syms))) than 4. Add more with @add_indices later."
        end
        if length(_frame_syms) < 4
            @warn "Manifold $($(name_symbol)): fewer frame indices " *
                  "($(length(_frame_syms))) than 4."
        end

        println("Defined manifold $($(name_symbol)) of dimension $(_dim)")

        # ── Register coordinate indices ──────────────────────────────────
        for _idx in _coord_syms
            register_coordinate_index!(_idx, $(tangent_symbol))
        end

        # ── Register frame indices ───────────────────────────────────────
        for _fidx in _frame_syms
            register_frame_index!(_fidx, $(tangent_symbol))
        end

        # ── Build index vectors for VBundle ──────────────────────────────
        local _t_coord = [CoordinateIndex(s, $(tangent_symbol))   for s in _coord_syms]
        local _c_coord = [CoordinateIndex(s, $(cotangent_symbol)) for s in _coord_syms]
        local _t_frame = [FrameIndex(s, $(tangent_symbol))   for s in _frame_syms]
        local _c_frame = [FrameIndex(s, $(cotangent_symbol)) for s in _frame_syms]

        # ── Register bundles ─────────────────────────────────────────────
        _VBUNDLES[$(tangent_symbol)] = VBundle(
            $(tangent_symbol), $(name_symbol), _dim, false,
            $(cotangent_symbol), _t_coord, _t_frame
        )
        _VBUNDLES[$(cotangent_symbol)] = VBundle(
            $(cotangent_symbol), $(name_symbol), _dim, true,
            $(tangent_symbol), _c_coord, _c_frame
        )

        # ── Register manifold ────────────────────────────────────────────
        _MANIFOLDS[$(name_symbol)] = Manifold(
            $(name_symbol), _dim,
            $(tangent_symbol), $(cotangent_symbol),
            [$(tangent_symbol), $(cotangent_symbol)]
        )

        # ── Bind variables in caller's scope ─────────────────────────────
        $(esc(name))           = _MANIFOLDS[$(name_symbol)]
        $(esc(tangent_name))   = _VBUNDLES[$(tangent_symbol)]
        $(esc(cotangent_name)) = _VBUNDLES[$(cotangent_symbol)]

        # CoordinateIndex variables
        $([ :($(esc(s)) = CoordinateIndex($(QuoteNode(s)), $(tangent_symbol))) for s in coord_symbols ]...)

        # FrameIndex variables
        $([ :($(esc(s)) = FrameIndex($(QuoteNode(s)), $(tangent_symbol))) for s in frame_symbols ]...)

        # ── Register coordinate frame ────────────────────────────────────
        $(_gen_coord_frame_registration_expr(
            tangent_symbol, cotangent_symbol, coord_frame, coord_coframe))

        # ── Register moving frame ────────────────────────────────────────
        $(_gen_moving_frame_registration_expr(
            frame_bundle_name, coframe_bundle_name,
            tangent_symbol, cotangent_symbol,
            moving_frame, moving_coframe,
            name_symbol,
        ))

        nothing
    end
end


# =========================================
# 7.  @undef_manifold macro
# =========================================

"""
    @undef_manifold <name>

Remove a manifold and all its associated bundles, indices, bases, and frame
bundles from the module-level registries.

After this call `_MANIFOLDS[name]`, `_VBUNDLES[tangentM]`,
`_VBUNDLES[cotangentM]`, all associated entries in `_COORDINATE_INDICES`,
`_FRAME_INDICES`, `_BASES`, and `_FRAME_BUNDLES` no longer exist.

The Julia variable `<name>` in the caller's scope still holds the old
`Manifold` struct (bindings cannot be deleted at runtime). Accessing any
field on it will trigger a stale-reference warning.
"""
macro undef_manifold(name)
    name isa Symbol ||
        error("@undef_manifold: argument must be a symbol, got $name")
    name_sym = QuoteNode(name)
    quote
        haskey(_MANIFOLDS, $(name_sym)) ||
            error(
                "@undef_manifold: manifold $($(name_sym)) is not registered. " *
                "Call @def_manifold $($(name_sym)) first."
            )

        local _m_data   = _MANIFOLDS[$(name_sym)]
        local _tb_name  = getfield(_m_data, :tangent_bundle)
        local _ctb_name = getfield(_m_data, :cotangent_bundle)

        if haskey(_VBUNDLES, _tb_name)
            local _tb_vb = _VBUNDLES[_tb_name]
            for _t_idx in getfield(_tb_vb, :coordinate_indices)
                unregister_index!(getfield(_t_idx, :symbol))
            end
            for _t_idx in getfield(_tb_vb, :frame_indices)
                unregister_index!(getfield(_t_idx, :symbol))
            end
            delete!(_VBUNDLES, _tb_name)
            delete!(_VBUNDLES, _ctb_name)
        end

        for _ftype in (:coordinate, :frame)
            delete!(_BASES, (_tb_name,  _ftype))
            delete!(_BASES, (_ctb_name, _ftype))
        end

        local _fb_name  = Symbol("frame",   $(name_sym))
        local _cfb_name = Symbol("coframe", $(name_sym))
        delete!(_FRAME_BUNDLES, _fb_name)
        delete!(_FRAME_BUNDLES, _cfb_name)

        delete!(_MANIFOLDS, $(name_sym))

        @warn "Manifold $($(name_sym)) has been undefined. " *
              "Variable still holds a stale reference. " *
              "Restart the kernel to fully clear the binding."
        println("Undefined tangent bundle:   $_tb_name")
        println("Undefined cotangent bundle: $_ctb_name")
        nothing
    end
end


# =========================================
# 8.  Stale-reference guards
# =========================================

function Base.getproperty(m::Manifold, field::Symbol)
    if !haskey(_MANIFOLDS, getfield(m, :name))
        @warn "Manifold :$(getfield(m, :name)) has been undefined. " *
              "Variable still holds a stale reference."
        return nothing
    end
    getfield(m, field)
end

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

function Base.propertynames(::VBundle, private::Bool=false)
    (:name, :manifold, :dim, :isdual, :dual, :coordinate_indices, :frame_indices, :bases)
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