# =========================================
# vbundles.jl — AbstractTensors.jl
#
# Design principles:
#   - Additional vector bundles beyond the canonical tangent/cotangent pair.
#     @def_vbundle binds E and dualE as variables in the caller's scope,
#     all queryable via dot access: E.isdual, E.indices, etc.
#   - All metadata lives in module-level registries (_VBUNDLES, _MANIFOLDS).
#   - Indices are registered via register_index! from indices.jl.
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

"""
    @def_vbundle name manifold dim [idx1, idx2, ...]

Define a new vector bundle `name` of fibre dimension `dim` over `manifold`,
and its dual bundle `dual<name>`. Bind the following variables in the
caller's scope:

- `name`       → a [`VBundle`](@ref) instance (`isdual = false`)
- `dualname`   → a [`VBundle`](@ref) instance (`isdual = true`)

Each index symbol in `[idx1, idx2, ...]` is bound to an [`IndexSymbol`](@ref)
and registered to `name` (the primal bundle). The dual bundle shares the
same index symbols — `down(v1)` resolves to `TensorIndex(:v1, :dualname)`
automatically via `down` / `dual_vbundle` (internal).

`dim` accepts a concrete integer or a bare symbol for parametric bundles.
The fibre dimension is independent of the base manifold dimension.

Registers both bundles in `_VBUNDLES` and appends their names to
`manifold.vbundles`.

### Examples
```julia
@def_manifold M 4 [a1, a2, a3, a4]
@def_vbundle E M 3 [A1, A2, A3]    # rank-3 bundle and its dual over M
@def_vbundle E M r [A1, A2, A3]    # parametric fibre dimension

E.isdual       # false
dualE.isdual   # true
v1.vbundle     # :E
up(A1)         # TensorIndex(:v1, :E)
down(A1)       # TensorIndex(:v1, :dualE)
M.vbundles     # [:tangentM, :cotangentM, :E, :dualE]
```
"""
macro def_vbundle(name, manifold_name, dim, indices)
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

    dual_name       = Symbol("dual", name)
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
            for _old_idx in getfield(_VBUNDLES[$(name_symbol)], :indices)
                unregister_index!(getfield(_old_idx, :symbol))
            end
            local _m_old = _MANIFOLDS[$(manifold_symbol)]
            filter!(
                x -> x ∉ ($(name_symbol), $(dual_symbol)),
                getfield(_m_old, :vbundles)
            )
            delete!(_VBUNDLES, $(name_symbol))
            delete!(_VBUNDLES, $(dual_symbol))
        end

        # ── Validate dimension ────────────────────────────────────────────
        local _dim::Dim = $(dim_expr)
        _dim isa Int && (_dim > 0 ||
            error("@def_vbundle: dimension must be positive, got $_dim"))

        # ── Step 1: Register indices ──────────────────────────────────────
        # Indices are registered to the primal bundle only.
        # dual_vbundle() resolves the dual at lookup time.
        for _idx in $(index_symbols)
            register_index!(_idx, $(name_symbol))
        end

        # ── Step 2: Build TensorIndex vectors ────────────────────────────
        local _p_indices = [TensorIndex(s, $(name_symbol)) for s in $(index_symbols)]
        local _d_indices = [TensorIndex(s, $(dual_symbol)) for s in $(index_symbols)]

        # ── Step 3: Register bundles in _VBUNDLES ────────────────────────
        _VBUNDLES[$(name_symbol)] = VBundle(
            $(name_symbol), $(manifold_symbol), _dim, false, _p_indices
        )
        _VBUNDLES[$(dual_symbol)] = VBundle(
            $(dual_symbol), $(manifold_symbol), _dim, true, _d_indices
        )

        # ── Step 4: Append to manifold's vbundles list ───────────────────
        local _m_data = _MANIFOLDS[$(manifold_symbol)]
        push!(getfield(_m_data, :vbundles), $(name_symbol))
        push!(getfield(_m_data, :vbundles), $(dual_symbol))

        # ── Step 5: Bind VBundle instances in caller's scope ─────────────
        $(esc(name))      = _VBUNDLES[$(name_symbol)]
        $(esc(dual_name)) = _VBUNDLES[$(dual_symbol)]

        # ── Step 6: Bind IndexSymbol variables in caller's scope ─────────
        $([ :($(esc(s)) = IndexSymbol($(QuoteNode(s)))) for s in index_symbols ]...)

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

        # ── Unregister indices ────────────────────────────────────────────
        for _idx in getfield(_vb_data, :indices)
            unregister_index!(getfield(_idx, :symbol))
        end

        # ── Remove from manifold's vbundles list ──────────────────────────
        if haskey(_MANIFOLDS, _man_sym)
            filter!(
                x -> x ∉ ($(name_symbol), $(dual_symbol)),
                getfield(_MANIFOLDS[_man_sym], :vbundles)
            )
        end

        # ── Delete from _VBUNDLES ─────────────────────────────────────────
        delete!(_VBUNDLES, $(name_symbol))
        delete!(_VBUNDLES, $(dual_symbol))

        @warn "VBundle $($(name_symbol)) and dual $($(dual_symbol)) have been undefined. " *
              "Variables still hold stale references. " *
              "Restart the kernel to fully clear the bindings."
        nothing
    end
end


# =========================================
# Exports
# =========================================

export @def_vbundle, @undef_vbundle