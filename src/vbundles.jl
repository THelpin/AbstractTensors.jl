# =========================================
# vbundles.jl — AbstractTensors.jl
#
# Design principles:
#   - Additional vector bundles beyond the canonical tangent/cotangent pair.
#     @def_vbundle binds E and dualE as variables in the caller's scope,
#     all queryable via dot access: E.isdual, E.indices, etc.
#   - All metadata lives in module-level registries (_VBUNDLES, _MANIFOLDS).
#   - Indices are registered via register_basis_index! from indices.jl.
#   - Fibre dimension is independent of the base manifold dimension.
#   - The stale-reference guard in Base.getproperty(::VBundle, ...) defined
#     in manifolds.jl applies to all VBundle instances, including those
#     created here.
#
# Depends on: types.jl, indices.jl, manifolds.jl
# =========================================


# =========================================
# 1.  @def_vbundle macro
# =========================================


# ── kwargs parser ─────────────────────────────────────────────────────────────

# Parse only dual_name= from @def_vbundle kwargs.
# Returns dual_name_override::Union{Symbol,Nothing}
# Note: basis= / cobasis= have been removed; coordinate frames are
# registered exclusively by @def_manifold's inline frame registration.
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
    @def_vbundle name manifold dim [idx1, idx2, ...]

Define a new vector bundle `name` of fibre dimension `dim` over `manifold`,
and its dual bundle `dual<name>`. Bind the following variables in the
caller's scope:

- `name`       → a [`VBundle`](@ref) instance (`isdual = false`)
- `dualname`   → a [`VBundle`](@ref) instance (`isdual = true`)

Each index symbol in `[idx1, idx2, ...]` is bound to a contravariant
[`BasisIndex`](@ref) and registered to `name` (the primal bundle). The dual
bundle shares the same index symbols — `-A1` resolves to
`BasisIndex(:A1, :dualname)` via [`flip`](@ref) and the `dual` field on
[`VBundle`](@ref).

`dim` accepts a concrete integer or a bare symbol for parametric bundles.
The fibre dimension is independent of the base manifold dimension.

Registers both bundles in `_VBUNDLES` and appends their names to
`manifold.vbundles`.

### Examples
```julia
@def_manifold M 4 [a1, a2, a3, a4] [B1, B2, B3, B4]
@def_vbundle E M 3 [A1, A2, A3]    # rank-3 bundle and its dual over M
@def_vbundle E M r [A1, A2, A3]    # parametric fibre dimension

E.isdual       # false
dualE.isdual   # true
A1.vbundle     # :E
-A1            # BasisIndex(:A1, :dualE)
M.vbundles     # [:tangentM, :cotangentM, :E, :dualE]
```
"""
macro def_vbundle(name, manifold_name, dim, indices, kwargs...)
    name isa Symbol ||
        error("@def_vbundle: first argument must be a bundle symbol, got $name")
    manifold_name isa Symbol ||
        error("@def_vbundle: second argument must be a manifold symbol, got $manifold_name")

    # Lift bare symbol to QuoteNode so it is treated as a symbolic dimension,
    # not a variable name to evaluate. Integer literals pass through directly.
    dim_expr = if dim isa Integer
        dim
    elseif dim isa Symbol
        QuoteNode(dim)
    else
        esc(dim)
    end

    # Parse kwargs (only dual_name= is supported now)
    dual_name_override = _parse_vbundle_kwargs(kwargs)

    dual_name       = isnothing(dual_name_override) ? Symbol("dual", name) : dual_name_override
    name_symbol     = QuoteNode(name)
    dual_symbol     = QuoteNode(dual_name)
    manifold_symbol = QuoteNode(manifold_name)
    index_symbols   = _macro_index_symbols(indices)

    quote
        # ── Guard: manifold must exist ────────────────────────────────────
        haskey(_MANIFOLDS, $(manifold_symbol)) ||
            error(
                "@def_vbundle: manifold $($(manifold_symbol)) is not registered. " *
                "Call @def_manifold $($(manifold_symbol)) first."
            )

        # ── Guard: warn and clean up if redefining ────────────────────────
        if haskey(_VBUNDLES, $(name_symbol))
            @warn "VBundle $($(name_symbol)) is already defined. Redefining."
            local _old_dual_redef = getfield(_VBUNDLES[$(name_symbol)], :dual)
            for _old_idx in getfield(_VBUNDLES[$(name_symbol)], :indices)
                unregister_index!(getfield(_old_idx, :symbol))
            end
            local _m_old = _MANIFOLDS[$(manifold_symbol)]
            filter!(
                x -> x ∉ ($(name_symbol), _old_dual_redef),
                getfield(_m_old, :vbundles)
            )
            # Clean all frame types from _BASES
            for _ftype in (:coordinate, :moving)
                delete!(_BASES, ($(name_symbol),  _ftype))
                delete!(_BASES, (_old_dual_redef, _ftype))
            end
            delete!(_VBUNDLES, $(name_symbol))
            delete!(_VBUNDLES, _old_dual_redef)
        end

        # ── Validate dimension ────────────────────────────────────────────
        local _dim::Dim = $(dim_expr)
        _dim isa Int && (_dim > 0 ||
            error("@def_vbundle: dimension must be positive, got $_dim"))

        # ── Step 1: Register indices ──────────────────────────────────────
        # Indices are registered to the primal bundle only.
        for _idx in $(index_symbols)
            register_basis_index!(_idx, $(name_symbol))
        end

        # ── Step 2: Build BasisIndex vectors ─────────────────────────────
        local _p_indices = [BasisIndex(s, $(name_symbol)) for s in $(index_symbols)]
        local _d_indices = [BasisIndex(s, $(dual_symbol)) for s in $(index_symbols)]

        # ── Step 3: Register bundles in _VBUNDLES ────────────────────────
        _VBUNDLES[$(name_symbol)] = VBundle(
            $(name_symbol), $(manifold_symbol), _dim, false,
            $(dual_symbol), _p_indices
        )
        _VBUNDLES[$(dual_symbol)] = VBundle(
            $(dual_symbol), $(manifold_symbol), _dim, true,
            $(name_symbol), _d_indices
        )

        # ── Step 4: Append to manifold's vbundles list ───────────────────
        local _m_data = _MANIFOLDS[$(manifold_symbol)]
        push!(getfield(_m_data, :vbundles), $(name_symbol))
        push!(getfield(_m_data, :vbundles), $(dual_symbol))

        # ── Step 5: Bind VBundle instances in caller's scope ─────────────
        $(esc(name))      = _VBUNDLES[$(name_symbol)]
        $(esc(dual_name)) = _VBUNDLES[$(dual_symbol)]

        # ── Step 6: Bind BasisIndex variables in caller's scope ──────────
        $([ :($(esc(s)) = BasisIndex($(QuoteNode(s)), $(name_symbol))) for s in index_symbols ]...)

        println("Defined VBundle $($(name_symbol)) (dim=$(_dim)) " *
                "and dual $($(dual_symbol)) over manifold $($(manifold_symbol))")
        nothing
    end
end


# =========================================
# 2.  @undef_vbundle macro
# =========================================

"""
    @undef_vbundle name

Remove a user-defined vector bundle `name` and its dual `dual<name>` from
the module-level registries.

After this call:
- `_VBUNDLES[name]` and `_VBUNDLES[dualname]` no longer exist
- all index symbols registered to `name` are removed from `_IDX_REGISTRY`
- `name` and `dualname` are removed from `manifold.vbundles`

!!! note "Cannot remove tangent or cotangent bundles"
    `@undef_vbundle` only removes bundles created by `@def_vbundle`.
    To remove the canonical tangent and cotangent bundles, use
    `@undef_manifold` instead.

## Stale variable warning

Julia module-level bindings cannot be deleted at runtime. The variables
`name` and `dualname` in the caller's scope still hold the old `VBundle`
structs after this call. Accessing any field on them will trigger the
stale-reference warning defined in `Base.getproperty(::VBundle, ...)`.

To fully clear the names from your session, restart the Julia kernel.

### Examples
```julia
@def_manifold M 4 [a1, a2, a3, a4]
@def_vbundle E M 3 [v1, v2, v3]
@undef_vbundle E M

E.isdual    # → Warning: VBundle :E has been undefined. Variable still holds a stale reference.
```
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
        # ── Guard: bundle must exist ──────────────────────────────────────
        haskey(_VBUNDLES, $(name_symbol)) ||
            error(
                "@undef_vbundle: bundle $($(name_symbol)) is not registered. " *
                "Call @def_vbundle $($(name_symbol)) first, or check _VBUNDLES."
            )

        # ── Guard: refuse to remove canonical bundles ─────────────────────
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

        # ── Capture data before deletion ──────────────────────────────────
        local _vb_data  = _VBUNDLES[$(name_symbol)]
        local _man_sym  = getfield(_vb_data, :manifold)
        local _vb_dual  = getfield(_vb_data, :dual)   # actual dual name

        # ── Unregister indices ────────────────────────────────────────────
        for _idx in getfield(_vb_data, :indices)
            unregister_index!(getfield(_idx, :symbol))
        end

        # ── Remove from manifold's vbundles list ──────────────────────────
        if haskey(_MANIFOLDS, _man_sym)
            filter!(
                x -> x ∉ ($(name_symbol), _vb_dual),
                getfield(_MANIFOLDS[_man_sym], :vbundles)
            )
        end

        # ── Clean up frames in _BASES ──────────────────────────────────────
        for _ftype in (:coordinate, :moving)
            delete!(_BASES, ($(name_symbol), _ftype))
            delete!(_BASES, (_vb_dual,       _ftype))
        end

        # ── Delete from _VBUNDLES ─────────────────────────────────────────
        delete!(_VBUNDLES, $(name_symbol))
        delete!(_VBUNDLES, _vb_dual)

        @warn "VBundle $($(name_symbol)) and dual $(_vb_dual) have been undefined. " *
              "Variables still hold stale references. " *
              "Restart the kernel to fully clear the bindings."
        nothing
    end
end

# =========================================
# 10. show methods
# =========================================

function Base.show(io::IO, v::VBundle)
    variance_label = v.isdual ? "cotangent" : "tangent"
    bases_str = isempty(v.bases) ? "none" :
        join(string.(v.bases), ", ")
    print(io, "VBundle($(v.name), $(variance_label), dual=$(v.dual), " *
              "manifold=$(v.manifold), dim=$(v.dim), bases=[$bases_str])")
end

function Base.show(io::IO, ::MIME"text/html", v::VBundle)
    idx_strings = map(v.indices) do ti
        sym = string(ti.symbol)
        is_down(ti) ? "-$(sym)" : "+$(sym)"
    end
    bases_html = if isempty(v.bases)
        "<i>none</i>"
    else
        join([
            "<code>$(b.name)</code> <span style=\"color:#666;\">($(b.type))</span>"
            for b in v.bases
        ], ", ")
    end
    variance_label = v.isdual ? "Dual (cotangent)" : "Standard (tangent)"
    print(io, """
    <div style="border:1px solid #ddd;padding:10px;border-radius:5px;background:#f4faff;">
        <h4 style="margin-top:0;">VBundle: <span style="color:#0d6efd;">$(v.name)</span></h4>
        <p>Base Manifold: <b>$(v.manifold)</b> | Rank: <b>$(v.dim)</b> | Type: <b>$(variance_label)</b></p>
        <div style="background:white;border:1px inset #eee;padding:5px;margin-bottom:5px;">
            <b>Indices:</b> $(join(idx_strings, ", "))
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