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
    tangentM.bases            # [Basis(:cf_M, :tangentM, :coordinate, "∂"),
                              #  Basis(:mf_M, :tangentM, :frame, "e")]

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
- `bases`              : [`Basis`](@ref) objects for the coordinate and frame bases
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

function _macro_symbol_vector(expr, context::String)::Vector{Symbol}
    Meta.isexpr(expr, :vect) ||
        error("$context: expected a vector literal like [cf_M, ccf_M, mf_M, mcf_M], got $expr")
    out = Symbol[]
    for a in expr.args
        if a isa Symbol
            push!(out, a)
        elseif a isa QuoteNode && a.value isa Symbol
            push!(out, a.value)
        else
            error("$context: each entry must be a plain symbol, got $a")
        end
    end
    return out
end

function _parse_print_as_entry(a, context::String)::String
    if a isa String
        return a
    elseif a isa Symbol
        return String(a)
    elseif a isa QuoteNode
        v = a.value
        v isa String && return v
        v isa Symbol && return String(v)
    elseif Meta.isexpr(a, :string) && length(a.args) == 1 && a.args[1] isa String
        return a.args[1]
    end
    error("$context: each print_as entry must be a string or symbol, got $a")
end

function _macro_print_as_vector(expr, context::String)::Vector{String}
    Meta.isexpr(expr, :vect) ||
        error(
            "$context: print_as must be a vector literal like " *
            "[\"∂\", \"dx\", \"e\", \"θ\"], got $expr"
        )
    return [_parse_print_as_entry(a, context) for a in expr.args]
end

# ── helper: register coordinate frame in _BASES and bind variables ──────────
function _gen_coord_frame_registration_expr(
    primal_q      :: QuoteNode,
    dual_q        :: QuoteNode,
    bind_cf       :: Symbol,
    bind_ccf      :: Symbol,
    print_cf      :: String,
    print_ccf     :: String,
    manifold_q    :: QuoteNode,
)
    bind_cf_q    = QuoteNode(bind_cf)
    bind_ccf_q   = QuoteNode(bind_ccf)
    print_cf_q   = QuoteNode(print_cf)
    print_ccf_q  = QuoteNode(print_ccf)
    primal_key   = QuoteNode((primal_q.value, :coordinate))
    dual_key     = QuoteNode((dual_q.value,   :coordinate))
    coord_msg    = QuoteNode(
        "Defined coordinate frame $(print_cf) (binding :$(bind_cf)) on $(primal_q.value) " *
        "and coordinate coframe $(print_ccf) (binding :$(bind_ccf)) on $(dual_q.value)"
    )
    quote
        _warn_and_register_basis_binding!($(bind_cf_q),  $(primal_q), :coordinate, $(manifold_q))
        _warn_and_register_basis_binding!($(bind_ccf_q), $(dual_q),   :coordinate, $(manifold_q))

        _BASES[$(primal_key)] = Basis($(bind_cf_q),  $(primal_q), :coordinate, $(print_cf_q))
        _BASES[$(dual_key)]   = Basis($(bind_ccf_q), $(dual_q),   :coordinate, $(print_ccf_q))
        $(esc(bind_cf))       = _BASES[$(primal_key)]
        $(esc(bind_ccf))      = _BASES[$(dual_key)]
        println($(coord_msg))
        nothing
    end
end

# ── helper: register moving frame in _BASES, create FrameBundles, bind vars ─
function _gen_moving_frame_registration_expr(
    frame_name    :: Symbol,
    coframe_name  :: Symbol,
    primal_q      :: QuoteNode,
    dual_q        :: QuoteNode,
    bind_mf       :: Symbol,
    bind_mcf      :: Symbol,
    print_mf      :: String,
    print_mcf     :: String,
    manifold_q    :: QuoteNode,
)
    bind_mf_q   = QuoteNode(bind_mf)
    bind_mcf_q  = QuoteNode(bind_mcf)
    print_mf_q  = QuoteNode(print_mf)
    print_mcf_q = QuoteNode(print_mcf)
    fn_q        = QuoteNode(frame_name)
    cfn_q       = QuoteNode(coframe_name)
    primal_key  = QuoteNode((primal_q.value, :frame))
    dual_key    = QuoteNode((dual_q.value,   :frame))
    frame_msg   = QuoteNode(
        "Defined frame bundle $(fn_q.value) (moving frame $(print_mf), binding :$(bind_mf)) " *
        "and coframe bundle $(cfn_q.value) (moving coframe $(print_mcf), binding :$(bind_mcf)) " *
        "over $(manifold_q.value)"
    )
    quote
        _warn_and_register_basis_binding!($(bind_mf_q),  $(primal_q), :frame, $(manifold_q))
        _warn_and_register_basis_binding!($(bind_mcf_q), $(dual_q),   :frame, $(manifold_q))

        _BASES[$(primal_key)] = Basis($(bind_mf_q),  $(primal_q), :frame, $(print_mf_q))
        _BASES[$(dual_key)]   = Basis($(bind_mcf_q), $(dual_q),   :frame, $(print_mcf_q))

        _FRAME_BUNDLES[$(fn_q)]  = FrameBundle($(fn_q),  $(primal_q), $(cfn_q), _BASES[$(primal_key)])
        _FRAME_BUNDLES[$(cfn_q)] = FrameBundle($(cfn_q), $(dual_q),   $(fn_q),  _BASES[$(dual_key)])

        $(esc(frame_name))   = _FRAME_BUNDLES[$(fn_q)]
        $(esc(coframe_name)) = _FRAME_BUNDLES[$(cfn_q)]
        $(esc(bind_mf))      = _BASES[$(primal_key)]
        $(esc(bind_mcf))     = _BASES[$(dual_key)]

        println($(frame_msg))
        nothing
    end
end

# ── kwargs parser ────────────────────────────────────────────────────────────
function _parse_manifold_kwargs(manifold_name::Symbol, kwargs)
    default_bindings = _default_manifold_frame_bindings(manifold_name)
    frame_bindings   = collect(default_bindings)
    print_as         = collect(_DEFAULT_PRINT_AS)
    frames_given     = false
    print_as_given    = false

    for kw in kwargs
        Meta.isexpr(kw, :(=), 2) ||
            error("@def_manifold: expected keyword=value argument, got: $kw")
        k, v = kw.args

        if k === :frames
            frame_bindings = _macro_symbol_vector(v, "@def_manifold frames")
            frames_given = true
        elseif k === :print_as
            print_as = _macro_print_as_vector(v, "@def_manifold print_as")
            print_as_given = true
        elseif k in (:coord_frame, :coord_coframe, :moving_frame, :moving_coframe,
                     :natural_frame, :natural_coframe)
            error(
                "@def_manifold: keyword :$k is no longer supported. " *
                "Use frames=[cf_$(manifold_name), ccf_$(manifold_name), mf_$(manifold_name), mcf_$(manifold_name)] " *
                "and print_as=[\"∂\", \"dx\", \"e\", \"θ\"]."
            )
        else
            error(
                "@def_manifold: unknown keyword :$k. " *
                "Supported: frames, print_as."
            )
        end
    end

    length(frame_bindings) == 4 ||
        error("@def_manifold: frames must have exactly 4 symbols; got $(length(frame_bindings))")
    length(print_as) == 4 ||
        error("@def_manifold: print_as must have exactly 4 labels; got $(length(print_as))")
    length(unique(frame_bindings)) == 4 ||
        error("@def_manifold: all four frame bindings must be distinct; got $frame_bindings")
    length(unique(print_as)) == 4 ||
        error("@def_manifold: all four print_as labels must be distinct; got $print_as")

    return (
        frame_bindings[1], frame_bindings[2], frame_bindings[3], frame_bindings[4],
        print_as[1], print_as[2], print_as[3], print_as[4],
        frames_given, print_as_given,
    )
end

"""
    @def_manifold <name> dim coord_indices frame_indices [kwargs...]

Define a new manifold and automatically create its tangent and cotangent
bundles, coordinate frames, and moving frames.

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
- coordinate frame bindings `cf_<name>`, `ccf_<name>` (default; display as `∂`, `dx`)
- moving frame bindings `mf_<name>`, `mcf_<name>` (default; display as `e`, `θ`)
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

| Keyword     | Default | Description |
|:------------|:--------|:------------|
| `frames`    | `[cf_<name>, ccf_<name>, mf_<name>, mcf_<name>]` | Caller-scope binding symbols (coord frame, coord coframe, moving frame, moving coframe) |
| `print_as`  | `["∂", "dx", "e", "θ"]` | Display label strings for the four bases |

All four entries in each vector must be distinct. Reusing a binding already
registered for another manifold triggers a warning.

### Examples

~~~julia
# Minimal — default bindings cf_M, ccf_M, mf_M, mcf_M; display ∂, dx, e, θ
@def_manifold M 4 [a1, a2, a3, a4] [A1, A2, A3, A4]
# Binds: M, tangentM, cotangentM, frameM, coframeM,
#        cf_M, ccf_M, mf_M, mcf_M,
#        a1..a4 (CoordinateIndex), A1..A4 (FrameIndex)
ccf_M[a1]   # displays as dx[a1]

# Parametric dimension
@def_manifold M d [a1, a2, a3, a4] [A1, A2, A3, A4]

# Custom bindings and display labels
@def_manifold M 4 [a1, a2, a3, a4] [A1, A2, A3, A4] \\
    frames=[:cfM, :ccfM, :mfM, :mcfM] \\
    print_as=["∂", "dx", "e", "θ"]
~~~
"""
macro def_manifold(name, dim, coord_indices, frame_indices, kwargs...)
    name isa Symbol ||
        error("@def_manifold: first argument must be a symbol, got $name")

    bind_cf, bind_ccf, bind_mf, bind_mcf, print_cf, print_ccf, print_mf, print_mcf, _, _ =
        _parse_manifold_kwargs(name, kwargs)

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
            _unregister_basis_bindings_for_vbundles!([_old_tb, _old_ctb])
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
            tangent_symbol, cotangent_symbol,
            bind_cf, bind_ccf,
            print_cf, print_ccf,
            name_symbol,
        ))

        # ── Register moving frame ────────────────────────────────────────
        $(_gen_moving_frame_registration_expr(
            frame_bundle_name, coframe_bundle_name,
            tangent_symbol, cotangent_symbol,
            bind_mf, bind_mcf,
            print_mf, print_mcf,
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

        _unregister_basis_bindings_for_vbundles!([_tb_name, _ctb_name])
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