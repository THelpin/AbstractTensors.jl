# =========================================
# manifolds.jl — AbstractTensors.jl
#
# Design principles:
#   - Manifolds are singleton types; dispatch is on Type{M}.
#   - All metadata lives in module-level registries.
#   - Indices are registered via register_index! from indices.jl.
#   - VBundleRecord.isdual is the single authoritative source for
#     bundle variance (false = tangent, true = cotangent/dual).
#     No naming conventions are relied upon for this.
#
# xTensor analogs:
#   $Manifolds              → _MANIFOLDS
#   ManifoldQ[M]            → is_manifold(M)
#   DimOfManifold[M]        → dim_of_manifold(M)
#   TangentBundleOfManifold → tangent_bundle_of(M)
#   IndicesOfVBundle        → indices_of_vbundle(vb::Symbol)
# =========================================

# Depends on indices.jl being loaded first.
# In AbstractTensors.jl: include("indices.jl") before include("manifolds.jl")

# =========================================
# 1. Registry data structures
# =========================================

"""
    ManifoldRecord

Metadata stored for each registered manifold.

Fields
------
- `name`             : manifold name, e.g. `:M`
- `dim`              : dimension
- `tangent_bundle`   : name of the tangent bundle, e.g. `:TangentM`
- `cotangent_bundle` : name of the cotangent (dual) bundle, e.g. `:CoTangentM`
- `vbundles`         : all associated bundle names (tangent + cotangent to start)
"""
struct ManifoldRecord
    name::Symbol
    dim::Int
    tangent_bundle::Symbol
    cotangent_bundle::Symbol
    vbundles::Vector{Symbol}
end

"""
    VBundleRecord

Metadata stored for each registered vector bundle.

Fields
------
- `name`     : bundle name, e.g. `:TangentM`
- `manifold` : base manifold name, e.g. `:M`
- `dim`      : fibre dimension
- `isdual`   : false = tangent (standard) bundle, true = cotangent (dual) bundle.
               This is the single authoritative source of bundle variance —
               no naming convention is relied upon.
- `indices`  : the `TensorIndex` objects living in this bundle
"""
struct VBundleRecord
    name::Symbol
    manifold::Symbol
    dim::Int
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
const _MANIFOLDS = Dict{Symbol, ManifoldRecord}()

"""
Registry of all defined vector bundles.
Key: bundle name as Symbol (e.g. `:TangentM`).
"""
const _VBUNDLES = Dict{Symbol, VBundleRecord}()


# =========================================
# 3. Abstract types
# =========================================

"""
    abstract type Manifold end

Root type for all manifold singleton types created by `@def_manifold`.

    @def_manifold M 4 [μ, ν, ρ, σ]
    # generates: struct M <: Manifold end

Dispatch on the type itself, never on an instance:

    dim_of_manifold(M)       # 4
    tangent_bundle_of(M)     # :TangentM
    is_manifold(M)           # true
    is_manifold(Float64)     # false
"""
abstract type Manifold end
abstract type VBundle end
abstract type TangentBundle   <: VBundle end
abstract type CoTangentBundle <: VBundle end


# =========================================
# 4. dual_bundle  — defined here because indices.jl calls it
#    but _VBUNDLES lives here.
# =========================================

"""
    dual_bundle(vb::Symbol) -> Symbol

Return the dual bundle of `vb`. Reads `isdual` from `_VBUNDLES` —
no naming convention is assumed.

    dual_bundle(:TangentM)    →  :CoTangentM
    dual_bundle(:CoTangentM)  →  :TangentM
"""
function dual_bundle(vb::Symbol)
    haskey(_VBUNDLES, vb) || error("Vbundle :$vb is not registered.")
    r = _VBUNDLES[vb]
    m = _MANIFOLDS[r.manifold]
    r.isdual ? m.tangent_bundle : m.cotangent_bundle
end


# =========================================
# 5. Generic accessor functions
# =========================================

is_manifold(::Type{<:Manifold}) = true
is_manifold(::Any)              = false

is_vbundle(::Type{<:VBundle}) = true
is_vbundle(::Any)              = false

is_tangent_bundle(::Type{<:TangentBundle})     = true
is_tangent_bundle(::Any)                       = false

is_cotangent_bundle(::Type{<:CoTangentBundle}) = true
is_cotangent_bundle(::Any)                     = false

# Registry-based variants (work on Symbol keys, no type needed)
is_tangent_bundle(vb::Symbol)   = haskey(_VBUNDLES, vb) && !_VBUNDLES[vb].isdual
is_cotangent_bundle(vb::Symbol) = haskey(_VBUNDLES, vb) &&  _VBUNDLES[vb].isdual

"""
    dim_of_manifold(M::Type{<:Manifold}) -> Int
"""
function dim_of_manifold(M::Type{<:Manifold})
    name = nameof(M)
    haskey(_MANIFOLDS, name) || error("Manifold $name is not registered.")
    _MANIFOLDS[name].dim
end

"""
    tangent_bundle_of(M::Type{<:Manifold}) -> Symbol
"""
function tangent_bundle_of(M::Type{<:Manifold})
    name = nameof(M)
    haskey(_MANIFOLDS, name) || error("Manifold $name is not registered.")
    _MANIFOLDS[name].tangent_bundle
end

"""
    cotangent_bundle_of(M::Type{<:Manifold}) -> Symbol
"""
function cotangent_bundle_of(M::Type{<:Manifold})
    name = nameof(M)
    haskey(_MANIFOLDS, name) || error("Manifold $name is not registered.")
    _MANIFOLDS[name].cotangent_bundle
end

"""
    vbundles_of(M::Type{<:Manifold}) -> Vector{Symbol}
"""
function vbundles_of(M::Type{<:Manifold})
    name = nameof(M)
    haskey(_MANIFOLDS, name) || error("Manifold $name is not registered.")
    _MANIFOLDS[name].vbundles
end

base_manifold(::Any) = error("Not a registered bundle type.")

"""
    indices_of_vbundle(VB::Type{<:VBundle}) -> Vector{TensorIndex}
"""
function indices_of_vbundle(VB::Type{<:VBundle})
    vb_name = nameof(VB)
    haskey(_VBUNDLES, vb_name) || error("Bundle $vb_name is not registered.")
    _VBUNDLES[vb_name].indices
end


# =========================================
# Helper: parse index symbols at macro expansion time
# =========================================

function _macro_index_symbols(indices_expr)::Vector{Symbol}
    Meta.isexpr(indices_expr, :vect) ||
        error("@def_manifold: indices must be a vector literal like [μ, ν, ρ, σ], got $indices_expr")
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
bundles, registering all indices to the tangent bundle.

# Example
```julia
@def_manifold M 4 [μ, ν, ρ, σ]
is_manifold(M)           # true
dim_of_manifold(M)       # 4
tangent_bundle_of(M)     # :TangentM
up(:μ)                   # TensorIndex(:μ, :TangentM)
down(:μ)                 # TensorIndex(:μ, :CoTangentM)
```
"""
macro def_manifold(name, dim, indices)
    name isa Symbol ||
        error("@def_manifold: first argument must be a symbol, got $name")

    tangent_name     = Symbol("Tangent",   name)
    cotangent_name   = Symbol("CoTangent", name)
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
        local _dim::Int = $(esc(dim))
        local _indices  = $(index_symbols)   # Vector{Symbol}, embedded at expansion time

        _dim > 0 || error("@def_manifold: dimension must be positive, got $_dim")

        if length(_indices) < _dim
            @warn "Manifold $($(name_symbol)): fewer indices ($(length(_indices))) " *
                  "than dim=$(_dim). Add more with @indices later."
        end

        # ── Step 1: Register indices into _IDX_REGISTRY first ───────────
        # Must happen before constructing TensorIndex objects, because
        # up() / down() call index_home_vbundle() which reads _IDX_REGISTRY.
        for _idx in _indices
            register_index!(_idx, $(tangent_symbol))
        end

        # ── Step 2: Build TensorIndex vectors from registry ──────────────
        # up() reads _IDX_REGISTRY → tangent bundle.
        # down() reads _IDX_REGISTRY → then dual_bundle → cotangent bundle.
        # Both are now safe because step 1 is complete.
        # NOTE: dual_bundle needs _VBUNDLES, so we build a bootstrap
        # cotangent TensorIndex directly here before _VBUNDLES is populated.
        local _t_indices = [TensorIndex(s, $(tangent_symbol))   for s in _indices]
        local _c_indices = [TensorIndex(s, $(cotangent_symbol)) for s in _indices]

        # ── Step 3: Register bundles ─────────────────────────────────────
        _VBUNDLES[$(tangent_symbol)] = VBundleRecord(
            $(tangent_symbol), $(name_symbol), _dim, false, _t_indices
        )
        _VBUNDLES[$(cotangent_symbol)] = VBundleRecord(
            $(cotangent_symbol), $(name_symbol), _dim, true, _c_indices
        )

        # ── Step 4: Register manifold ────────────────────────────────────
        _MANIFOLDS[$(name_symbol)] = ManifoldRecord(
            $(name_symbol), _dim,
            $(tangent_symbol), $(cotangent_symbol),
            [$(tangent_symbol), $(cotangent_symbol)]
        )

        # ── Step 5: Define singleton types ───────────────────────────────
        struct $(esc(name))           <: Manifold       end
        struct $(esc(tangent_name))   <: TangentBundle   end
        struct $(esc(cotangent_name)) <: CoTangentBundle end

        # ── Step 6: Concrete dispatch methods ────────────────────────────
        AbstractTensors.is_manifold(::Type{$(esc(name))})                  = true
        AbstractTensors.is_tangent_bundle(::Type{$(esc(tangent_name))})    = true
        AbstractTensors.is_cotangent_bundle(::Type{$(esc(cotangent_name))}) = true
        AbstractTensors.dim_of_manifold(::Type{$(esc(name))})              = $(esc(dim))
        AbstractTensors.tangent_bundle_of(::Type{$(esc(name))})            = $(tangent_symbol)
        AbstractTensors.cotangent_bundle_of(::Type{$(esc(name))})          = $(cotangent_symbol)
        AbstractTensors.base_manifold(::Type{$(esc(tangent_name))})        = $(esc(name))
        AbstractTensors.base_manifold(::Type{$(esc(cotangent_name))})      = $(esc(name))

        println("Defined manifold $($(name_symbol)) of dimension $(_dim) " *
                "with tangent bundle $($(tangent_symbol)) " *
                "and cotangent bundle $($(cotangent_symbol))")
        nothing
    end
end


# =========================================
# 7. @undef_manifold macro
# =========================================

"""
    @undef_manifold name

Remove a manifold and all its associated bundles and index registrations.
"""
macro undef_manifold(name)
    name isa Symbol ||
        error("@undef_manifold: argument must be a symbol, got $name")

    name_sym = QuoteNode(name)

    quote
        haskey(_MANIFOLDS, $(name_sym)) ||
            error(
                "@undef_manifold: manifold $($(name_sym)) is not registered. " *
                "Call @def_manifold $($(name_sym)) first, or check list_manifolds()."
            )

        local _m_data  = _MANIFOLDS[$(name_sym)]
        local _tb_name = _m_data.tangent_bundle

        # Unregister all indices (stored in the tangent bundle record)
        if haskey(_VBUNDLES, _tb_name)
            for _t_idx in _VBUNDLES[_tb_name].indices
                unregister_index!(_t_idx.symbol)   # flat field, no .index.symbol
            end
            delete!(_VBUNDLES, _tb_name)
            delete!(_VBUNDLES, _m_data.cotangent_bundle)
        end

        delete!(_MANIFOLDS, $(name_sym))

        println("Undefined manifold:        $($(name_sym))")
        println("Undefined tangent bundle:  $(_m_data.tangent_bundle)")
        println("Undefined cotangent bundle: $(_m_data.cotangent_bundle)")
        nothing
    end
end


# =========================================
# 8. Utility / introspection
# =========================================

list_manifolds() = collect(keys(_MANIFOLDS))

function manifold_info(M::Type{<:Manifold})
    name = nameof(M)
    haskey(_MANIFOLDS, name) || error("Manifold $name is not registered.")
    d = _MANIFOLDS[name]
    tb_indices = _VBUNDLES[d.tangent_bundle].indices
    symbols = [idx.symbol for idx in tb_indices]   # flat field

    println("Manifold: $(d.name)")
    println("  Dimension:        $(d.dim)")
    println("  Tangent bundle:   $(d.tangent_bundle)")
    println("  Cotangent bundle: $(d.cotangent_bundle)")
    println("  Vbundles:         $(d.vbundles)")
    println("  Index Symbols:    $(symbols)")
end


# =========================================
# 9. show methods
# =========================================

function Base.show(io::IO, d::ManifoldRecord)
    print(io, "Manifold($(d.name), dim=$(d.dim), " *
              "TBundle=$(d.tangent_bundle), CBundle=$(d.cotangent_bundle))")
end

function Base.show(io::IO, ::MIME"text/html", d::ManifoldRecord)
    print(io, """
    <div style="border:1px solid #ddd;padding:10px;border-radius:5px;background:#f9f9f9;">
        <h4 style="margin-top:0;">Manifold: <span style="color:#d63384;">$(d.name)</span></h4>
        <table style="width:100%;border-collapse:collapse;">
            <tr><td style="font-weight:bold;width:150px;">Dimension</td><td>$(d.dim)</td></tr>
            <tr><td style="font-weight:bold;">Tangent Bundle</td><td><code>$(d.tangent_bundle)</code></td></tr>
            <tr><td style="font-weight:bold;">Cotangent Bundle</td><td><code>$(d.cotangent_bundle)</code></td></tr>
            <tr><td style="font-weight:bold;">All VBundles</td>
                <td>$(join(map(x -> "<code>$x</code>", d.vbundles), ", "))</td></tr>
        </table>
    </div>
    """)
end

function Base.show(io::IO, ::MIME"text/html", v::VBundleRecord)
    idx_strings = map(v.indices) do ti
        sym = string(ti.symbol)                     # flat field
        v.isdual ? "$(sym)&darr;" : "$(sym)&uarr;"  # isdual drives the arrow
    end
    variance_label = v.isdual ? "Dual (cotangent)" : "Standard (tangent)"
    print(io, """
    <div style="border:1px solid #ddd;padding:10px;border-radius:5px;background:#f4faff;">
        <h4 style="margin-top:0;">Vector Bundle: <span style="color:#0d6efd;">$(v.name)</span></h4>
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

export Manifold, VBundle, TangentBundle, CoTangentBundle
export ManifoldRecord, VBundleRecord
export _MANIFOLDS, _VBUNDLES
export dual_bundle, dual_vbundles
export is_manifold, is_vbundle, is_tangent_bundle, is_cotangent_bundle
export dim_of_manifold, tangent_bundle_of, cotangent_bundle_of, vbundles_of, base_manifold
export indices_of_vbundle
export list_manifolds, manifold_info
export @def_manifold, @undef_manifold