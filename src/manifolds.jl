# =========================================
# manifolds.jl — AbstractTensors.jl
#
# Design principles:
#   - Manifolds and vector bundles are plain struct instances.
#     @def_manifold binds M, tangentM, cotangentM as variables
#     in the caller's scope, all queryable via dot access:
#       M.dim, M.tangent_bundle, TangentM.isdual, etc.
#   - All metadata lives in module-level registries.
#   - Indices are registered via register_index! from indices.jl.
#   - VBundle.isdual is the single authoritative source for
#     bundle variance (false = tangent, true = cotangent/dual).
#     No naming conventions are relied upon for this.
#
# xTensor analogs:
#   $Manifolds              → _MANIFOLDS
#   ManifoldQ[M]            → is_manifold(M)
#   DimOfManifold[M]        → M.dim
#   TangentBundleOfManifold → M.tangent_bundle
#   IndicesOfVBundle        → vb.indices  (VBundle instance)
# =========================================

# Depends on indices.jl being loaded first.
# In AbstractTensors.jl: include("indices.jl") before include("manifolds.jl")

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
    M.vbundles          # [:TangentM, :CoTangentM]

### Fields

- `name`             : manifold name, e.g. `:M`
- `dim`              : dimension
- `tangent_bundle`   : name of the tangent bundle, e.g. `:tangentM`
- `cotangent_bundle` : name of the cotangent (dual) bundle, e.g. `:cotangentM`
- `vbundles`         : all associated bundle names
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
    tangentM.indices   # [TensorIndex(:a1, :TangentM), ...]

### Fields

- `name`     : bundle name, e.g. `:TangentM`
- `manifold` : base manifold name, e.g. `:M`
- `dim`      : fibre dimension
- `isdual`   : false = tangent (standard) bundle, true = cotangent (dual) bundle.
               This is the single authoritative source of bundle variance —
               no naming convention is relied upon.
- `indices`  : the `TensorIndex` objects living in this bundle
"""
struct VBundle
    name::Symbol
    manifold::Symbol
    dim::Dim
    isdual::Bool
    indices::Vector{TensorIndex}
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
Key: bundle name as Symbol (e.g. `:TangentM`).
"""
const _VBUNDLES = Dict{Symbol, VBundle}()


# =========================================
# 3. dual_vbundle — defined here because indices.jl calls it
#    but _VBUNDLES lives here.
# =========================================

"""
    dual_vbundle(vb::Symbol) -> Symbol

!!! warning "Internal"
    This function is intended for internal use by the AbstractTensors.jl
    package. It is not part of the public API and may change without notice.

Return the dual bundle of `vb`. Reads `isdual` from `_VBUNDLES` —
no naming convention is assumed.

    dual_vbundle(:tangentM)    →  :cotangentM
    dual_vbundle(:cotangentM)  →  :tangentM
"""
function dual_vbundle(vb::Symbol)
    haskey(_VBUNDLES, vb) || error("VBundle :$vb is not registered.")
    r = _VBUNDLES[vb]
    m = _MANIFOLDS[r.manifold]
    r.isdual ? m.tangent_bundle : m.cotangent_bundle
end


# =========================================
# 4. Predicates
# =========================================

# Internal predicates (not exported)
is_manifold(x) = x isa Manifold
is_vbundle(x) = x isa VBundle
is_tangent_bundle(x::VBundle) = !x.isdual
is_tangent_bundle(vb::Symbol) = haskey(_VBUNDLES, vb) && !_VBUNDLES[vb].isdual
is_tangent_bundle(::Any)      = false
is_cotangent_bundle(x::VBundle) = x.isdual
is_cotangent_bundle(vb::Symbol) = haskey(_VBUNDLES, vb) && _VBUNDLES[vb].isdual
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

"""
    @def_manifold name dim [idx1, idx2, ...]

Define a new manifold and automatically create its tangent and cotangent
bundles. Bind the following variables in the caller's scope:
- `name`            → a [`Manifold`](@ref) instance
- `tangent<name>`   → a [`VBundle`](@ref) instance (`isdual = false`)
- `cotangent<name>` → a [`VBundle`](@ref) instance (`isdual = true`)

Each index symbol is also bound to an [`IndexSymbol`](@ref) in the caller's scope.

`dim` can be a concrete integer or a symbolic name for parametric manifolds:

Register `name` in `_MANIFOLDS`, the tangent and cotangent bundles in
`_VBUNDLES`, and all index symbols in `_INDICES`.

#### Examples
```julia
@def_manifold M 4 [a1, a2, a3, a4]   # concrete dimension
@def_manifold M d [b1, b2, b3, b4]  # parametric dimension
```
"""
macro def_manifold(name, dim, indices)
    name isa Symbol ||
        error("@def_manifold: first argument must be a symbol, got $name")
    # If dim is a plain Symbol at expansion time, treat it as a symbolic
    # dimension directly — do not try to evaluate it as a variable.
    # If it is an integer literal or an expression, evaluate it normally.
    dim_expr = if dim isa Integer
        dim                  # integer literal — use directly
    elseif dim isa Symbol
        QuoteNode(dim)       # bare symbol like d or n — lift to :d, :n
    else
        esc(dim)             # expression — evaluate in caller's scope
    end
    tangent_name     = Symbol("tangent",   name)
    cotangent_name   = Symbol("cotangent", name)
    name_symbol      = QuoteNode(name)
    tangent_symbol   = QuoteNode(tangent_name)
    cotangent_symbol = QuoteNode(cotangent_name)
    index_symbols    = _macro_index_symbols(indices)

    quote
        # ── Guard: clean up stale registry entries if redefining ─────────
        if haskey(_MANIFOLDS, $(name_symbol))
            @warn "Manifold $($(name_symbol)) is already defined. Redefining."
            local _old_tb = _MANIFOLDS[$(name_symbol)].tangent_bundle
            if haskey(_VBUNDLES, _old_tb)
                for _old_idx in _VBUNDLES[_old_tb].indices
                    unregister_index!(_old_idx.symbol)
                end
                delete!(_VBUNDLES, _old_tb)
                delete!(_VBUNDLES, _MANIFOLDS[$(name_symbol)].cotangent_bundle)
            end
            delete!(_MANIFOLDS, $(name_symbol))
        end

        # ── Runtime locals ───────────────────────────────────────────────
        local _dim::Dim = $(dim_expr)
        local _indices  = $(index_symbols)   # Vector{Symbol}, embedded at expansion time

        _dim isa Int && (_dim > 0 || error("@def_manifold: dimension must be positive, got $_dim"))

        if length(_indices) < 4
            @warn "Manifold $($(name_symbol)): fewer indices ($(length(_indices))) " *
                  "than 4. Add more with @add_indices later."
        end

        # ── Step 1: Register indices into _INDICES first ──────────────────
        # Must happen before constructing TensorIndex objects, because
        # up() / down() call index_home_vbundle() which reads _INDICES.
        for _idx in _indices
            register_index!(_idx, $(tangent_symbol))
        end

        # ── Step 2: Build TensorIndex vectors ────────────────────────────
        # Built directly with flat constructor — dual_vbundle is not yet
        # available because _VBUNDLES is not yet populated.
        local _t_indices = [TensorIndex(s, $(tangent_symbol))   for s in _indices]
        local _c_indices = [TensorIndex(s, $(cotangent_symbol)) for s in _indices]

        # ── Step 3: Register bundles ─────────────────────────────────────
        _VBUNDLES[$(tangent_symbol)] = VBundle(
            $(tangent_symbol), $(name_symbol), _dim, false, _t_indices
        )
        _VBUNDLES[$(cotangent_symbol)] = VBundle(
            $(cotangent_symbol), $(name_symbol), _dim, true, _c_indices
        )

        # ── Step 4: Register manifold ────────────────────────────────────
        _MANIFOLDS[$(name_symbol)] = Manifold(
            $(name_symbol), _dim,
            $(tangent_symbol), $(cotangent_symbol),
            [$(tangent_symbol), $(cotangent_symbol)]
        )

        # ── Step 5: Bind Manifold and VBundle instances in caller's scope ─
        $(esc(name))          = _MANIFOLDS[$(name_symbol)]
        $(esc(tangent_name))   = _VBUNDLES[$(tangent_symbol)]
        $(esc(cotangent_name)) = _VBUNDLES[$(cotangent_symbol)]

        # ── Step 6: Bind IndexSymbol variables in caller's scope ─────────
        # Each index symbol gets a variable bound to an IndexSymbol object,
        # enabling property-style access (a1.vbundle) and direct use with
        # up() / down() without the colon prefix.
        $([ :($(esc(s)) = IndexSymbol($(QuoteNode(s)))) for s in index_symbols ]...)

        println("Defined manifold $($(name_symbol)) of dimension $(_dim) " *
                "with tangent bundle $($(tangent_symbol)) " *
                "and cotangent bundle $($(cotangent_symbol))")
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
- every index symbol that belonged to the tangent bundle is unregistered
  from `_INDICES`

## Stale variable warning

Julia module-level bindings cannot be deleted at runtime. The variable
`name` in the caller's scope will still exist and still hold the old
`Manifold` struct after this call. Attempting to access any field on
that stale reference will raise an immediate error:

```julia
@def_manifold M 4 [a1, a2, a3, a4]
@undef_manifold M

M.dim   # → Warining: Manifold :M has been undefined. Variable still holds a stale reference.
```

This is enforced by `Base.getproperty(::Manifold, ...)`, which checks
registry membership before every field access. The same guard is applied
to `VBundle` via `Base.getproperty(::VBundle, ...)`.

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
            for _t_idx in getfield(_VBUNDLES[_tb_name], :indices)
                unregister_index!(getfield(_t_idx, :symbol))
            end
            delete!(_VBUNDLES, _tb_name)
            delete!(_VBUNDLES, _ctb_name)
        end

        delete!(_MANIFOLDS, $(name_sym))

        # Now safe — we use the captured Symbols, not the stale struct.
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
descriptive error:

```julia
@def_manifold M 4 [a1, a2, a3, a4]
@undef_manifold M
M.dim   # → ERROR: Manifold :M has been undefined. Variable still holds a stale reference.
```

!!! note
    Direct `getfield` calls bypass this guard. All user-facing access
    should go through dot syntax or the accessor functions.
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
registered in `_VBUNDLES` before returning the requested field. Stale
`VBundle` variables (e.g. `TangentM` after `@undef_manifold M`) raise an
immediate error rather than silently returning outdated data.

```julia
@def_manifold M 4 [a1, a2, a3, a4]
@undef_manifold M
tangentM.isdual   # → ERROR: VBundle :tangentM has been undefined. Variable still holds a stale reference.
```
"""

function Base.getproperty(v::VBundle, field::Symbol)
    if !haskey(_VBUNDLES, getfield(v, :name))
        @warn "VBundle :$(getfield(v, :name)) has been undefined. " *
              "Variable still holds a stale reference."
        return nothing
    end
    getfield(v, field)
end

# =========================================
# 8. Utility / introspection
# =========================================

"""
    list_manifolds() -> Vector{Symbol}

!!! warning "Internal"
    Return the names of all currently registered manifolds.
"""
list_manifolds() = collect(keys(_MANIFOLDS))

# =========================================
# 9. show methods
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

function Base.show(io::IO, v::VBundle)
    variance_label = v.isdual ? "cotangent" : "tangent"
    print(io, "VBundle($(v.name), $(variance_label), manifold=$(v.manifold), dim=$(v.dim))")
end

function Base.show(io::IO, ::MIME"text/html", v::VBundle)
    idx_strings = map(v.indices) do ti
        sym = string(ti.symbol)
        v.isdual ? "$(sym)&darr;" : "$(sym)&uarr;"
    end
    variance_label = v.isdual ? "Dual (cotangent)" : "Standard (tangent)"
    print(io, """
    <div style="border:1px solid #ddd;padding:10px;border-radius:5px;background:#f4faff;">
        <h4 style="margin-top:0;">VBundle: <span style="color:#0d6efd;">$(v.name)</span></h4>
        <p>Base Manifold: <b>$(v.manifold)</b> | Rank: <b>$(v.dim)</b> | Type: <b>$(variance_label)</b></p>
        <div style="background:white;border:1px inset #eee;padding:5px;">
            <b>Indices:</b> $(join(idx_strings, ", "))
        </div>
    </div>
    """)
end

"""
    show_registry()

Print an HTML summary of all registered manifolds and bundles.
Note: requires an IJulia / Pluto context for HTML rendering.
"""
function show_registry()
    for (_, rec) in _MANIFOLDS
        display(rec)
    end
    for (_, rec) in _VBUNDLES
        display(rec)
    end
end


# =========================================
# Exports
# =========================================

export Manifold, VBundle
export _MANIFOLDS, _VBUNDLES 
export @def_manifold, @undef_manifold