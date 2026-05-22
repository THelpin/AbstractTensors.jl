# =========================================
# vbundles.jl — SymbolicTensors.jl
#
# Additional vector bundles beyond the canonical tangent/cotangent pair.
# @def_vbundle binds E and dualE as variables in the caller's scope.
# Indices are registered via register_frame_index! from indices.jl.
#
# Depends on: types.jl, indices.jl, manifolds.jl
# =========================================


# =========================================
# 1.  @def_vbundle macro
# =========================================

function _parse_vbundle_kwargs(kwargs)
    dual_name_override = nothing
    for kw in kwargs
        Meta.isexpr(kw, :(=), 2) ||
            error("@def_vbundle: expected keyword=value argument, got: $kw")
        k, v = kw.args
        sym_val = if v isa Symbol
            v
        elseif v isa QuoteNode && v.value isa Symbol
            v.value
        else
            nothing
        end
        if k === :dual_name
            sym_val !== nothing ||
                error("@def_vbundle: dual_name must be a symbol, got $v")
            dual_name_override = sym_val
        else
            error(
                "@def_vbundle: unknown keyword :$k. " *
                "Supported: dual_name."
            )
        end
    end
    return dual_name_override
end

"""
    @def_vbundle <name> manifold dim [idx1, idx2, ...]

Define a new vector bundle `<name>` of fibre dimension `dim` over `manifold`,
and its dual bundle `dual<name>`. Bind the following variables in the
caller's scope:

- `<name>`      → a [`VBundle`](@ref) instance (`isdual = false`)
- `dual<name>`  → a [`VBundle`](@ref) instance (`isdual = true`)

Each index symbol in `[idx1, idx2, ...]` is bound to a contravariant
[`FrameIndex`](@ref) and registered to `<name>` (the primal bundle).

`dim` accepts a concrete integer or a bare symbol for parametric bundles.

### Examples

~~~julia
@def_manifold M 4 [a1, a2, a3, a4] [A1, A2, A3, A4]
@def_vbundle E M 3 [v1, v2, v3]    # rank-3 bundle and its dual over M

E.isdual       # false
dualE.isdual   # true
v1.vbundle     # :E
-v1            # FrameIndex(:v1, :dualE)
~~~
"""
macro def_vbundle(name, manifold_name, dim, indices, kwargs...)
    name isa Symbol ||
        error("@def_vbundle: first argument must be a bundle symbol, got $name")
    manifold_name isa Symbol ||
        error("@def_vbundle: second argument must be a manifold symbol, got $manifold_name")

    dim_expr = if dim isa Integer
        dim
    elseif dim isa Symbol
        QuoteNode(dim)
    else
        esc(dim)
    end

    dual_name_override = _parse_vbundle_kwargs(kwargs)
    dual_name       = isnothing(dual_name_override) ? Symbol("dual", name) : dual_name_override
    name_symbol     = QuoteNode(name)
    dual_symbol     = QuoteNode(dual_name)
    manifold_symbol = QuoteNode(manifold_name)
    index_symbols   = _macro_index_symbols(indices)

    quote
        haskey(_MANIFOLDS, $(manifold_symbol)) ||
            error(
                "@def_vbundle: manifold $($(manifold_symbol)) is not registered. " *
                "Call @def_manifold $($(manifold_symbol)) first."
            )

        if haskey(_VBUNDLES, $(name_symbol))
            @warn "VBundle $($(name_symbol)) is already defined. Redefining."
            local _old_dual_redef = getfield(_VBUNDLES[$(name_symbol)], :dual)
            for _old_idx in getfield(_VBUNDLES[$(name_symbol)], :frame_indices)
                unregister_index!(getfield(_old_idx, :symbol))
            end
            local _m_old = _MANIFOLDS[$(manifold_symbol)]
            filter!(
                x -> x ∉ ($(name_symbol), _old_dual_redef),
                getfield(_m_old, :vbundles)
            )
            for _ftype in (:coordinate, :frame)
                delete!(_BASES, ($(name_symbol),  _ftype))
                delete!(_BASES, (_old_dual_redef, _ftype))
            end
            delete!(_VBUNDLES, $(name_symbol))
            delete!(_VBUNDLES, _old_dual_redef)
        end

        local _dim::Dim = $(dim_expr)
        _dim isa Int && (_dim > 0 ||
            error("@def_vbundle: dimension must be positive, got $_dim"))

        for _idx in $(index_symbols)
            register_frame_index!(_idx, $(name_symbol))
        end

        local _p_indices = [FrameIndex(s, $(name_symbol)) for s in $(index_symbols)]
        local _d_indices = [FrameIndex(s, $(dual_symbol)) for s in $(index_symbols)]

        _VBUNDLES[$(name_symbol)] = VBundle(
            $(name_symbol), $(manifold_symbol), _dim, false,
            $(dual_symbol), CoordinateIndex[], _p_indices
        )
        _VBUNDLES[$(dual_symbol)] = VBundle(
            $(dual_symbol), $(manifold_symbol), _dim, true,
            $(name_symbol), CoordinateIndex[], _d_indices
        )

        local _m_data = _MANIFOLDS[$(manifold_symbol)]
        push!(getfield(_m_data, :vbundles), $(name_symbol))
        push!(getfield(_m_data, :vbundles), $(dual_symbol))

        $(esc(name))      = _VBUNDLES[$(name_symbol)]
        $(esc(dual_name)) = _VBUNDLES[$(dual_symbol)]

        $([ :($(esc(s)) = FrameIndex($(QuoteNode(s)), $(name_symbol))) for s in index_symbols ]...)

        println("Defined VBundle $($(name_symbol)) (dim=$(_dim)) " *
                "and dual $($(dual_symbol)) over manifold $($(manifold_symbol))")
        nothing
    end
end


# =========================================
# 2.  @undef_vbundle macro
# =========================================

"""
    @undef_vbundle <name> manifold

Remove a user-defined vector bundle `<name>` and its dual `dual<name>` from
the module-level registries.

Cannot remove canonical tangent or cotangent bundles — use
`@undef_manifold` for those.
"""
macro undef_vbundle(name, manifold_name)
    name isa Symbol ||
        error("@undef_vbundle: first argument must be a bundle symbol, got $name")
    manifold_name isa Symbol ||
        error("@undef_vbundle: second argument must be a manifold symbol, got $manifold_name")

    name_symbol     = QuoteNode(name)
    dual_symbol     = QuoteNode(Symbol("dual", name))
    manifold_symbol = QuoteNode(manifold_name)

    quote
        haskey(_VBUNDLES, $(name_symbol)) ||
            error(
                "@undef_vbundle: bundle $($(name_symbol)) is not registered."
            )

        if haskey(_MANIFOLDS, $(manifold_symbol))
            local _m     = _MANIFOLDS[$(manifold_symbol)]
            local _tb    = getfield(_m, :tangent_bundle)
            local _ctb   = getfield(_m, :cotangent_bundle)
            $(name_symbol) ∈ (_tb, _ctb) &&
                error(
                    "@undef_vbundle: $($(name_symbol)) is a canonical bundle of " *
                    "manifold $($(manifold_symbol)). Use @undef_manifold to remove it."
                )
        end

        local _vb_data  = _VBUNDLES[$(name_symbol)]
        local _man_sym  = getfield(_vb_data, :manifold)
        local _vb_dual  = getfield(_vb_data, :dual)

        for _idx in getfield(_vb_data, :frame_indices)
            unregister_index!(getfield(_idx, :symbol))
        end

        if haskey(_MANIFOLDS, _man_sym)
            filter!(
                x -> x ∉ ($(name_symbol), _vb_dual),
                getfield(_MANIFOLDS[_man_sym], :vbundles)
            )
        end

        for _ftype in (:coordinate, :frame)
            delete!(_BASES, ($(name_symbol), _ftype))
            delete!(_BASES, (_vb_dual,       _ftype))
        end

        delete!(_VBUNDLES, $(name_symbol))
        delete!(_VBUNDLES, _vb_dual)

        @warn "VBundle $($(name_symbol)) and dual $(_vb_dual) have been undefined. " *
              "Variables still hold stale references."
        nothing
    end
end


# =========================================
# 3. show methods
# =========================================

function Base.show(io::IO, ::MIME"text/plain", v::VBundle)
    variance_label = v.isdual ? "cotangent" : "tangent"
    bases_str = isempty(v.bases) ? "none" : join(string.(v.bases), ", ")
    print(io, "VBundle($(v.name), $(variance_label), dual=$(v.dual), " *
              "manifold=$(v.manifold), dim=$(v.dim), bases=[$bases_str])")
end

function _index_strings(idxs)
    map(idxs) do ti
        sym = string(ti.symbol)
        is_down(ti) ? "-$(sym)" : "+$(sym)"
    end
end

function Base.show(io::IO, ::MIME"text/html", v::VBundle)
    coord_strings = _index_strings(v.coordinate_indices)
    frame_strings = _index_strings(v.frame_indices)
    bases_html = if isempty(v.bases)
        "<i>none</i>"
    else
        join([
            "<code>$(b.print_as)</code> <span style=\"color:#666;\">($(b.type))</span>"
            for b in v.bases
        ], ", ")
    end
    variance_label = v.isdual ? "Dual (cotangent)" : "Standard (tangent)"
    coord_html = isempty(coord_strings) ? "<i>none</i>" : join(coord_strings, ", ")
    frame_html = isempty(frame_strings) ? "<i>none</i>" : join(frame_strings, ", ")
    print(io, """
    <div style="border:1px solid #ddd;padding:10px;border-radius:5px;background:#f4faff;">
        <h4 style="margin-top:0;">VBundle: <span style="color:#0d6efd;">$(v.name)</span></h4>
        <p>Base Manifold: <b>$(v.manifold)</b> | Rank: <b>$(v.dim)</b> | Type: <b>$(variance_label)</b></p>
        <div style="background:white;border:1px inset #eee;padding:5px;margin-bottom:5px;">
            <b>Coordinate indices:</b> $(coord_html)
        </div>
        <div style="background:white;border:1px inset #eee;padding:5px;margin-bottom:5px;">
            <b>Frame indices:</b> $(frame_html)
        </div>
        <div style="background:white;border:1px inset #eee;padding:5px;">
            <b>Bases:</b> $(bases_html)
        </div>
    </div>
    """)
end


# =========================================
# Exports
# =========================================

export @def_vbundle, @undef_vbundle