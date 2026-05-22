# =========================================
# frames.jl — SymbolicTensors.jl
#
# Frame bundles: coordinate frames (∂/dx) and arbitrary local frames (e/θ).
#
# Each vbundle may carry two basis types:
#
#   :coordinate — coordinate-induced frame/coframe when geometrically available
#                 registered by @def_manifold as part of vbundle creation
#   :frame      — arbitrary chosen local frame/coframe
#                 registered by @def_frame_bundle and bound as FrameBundle variables
#
# Registry:
#   _BASES[(vbundle, :coordinate)] → Basis(:ccf_M, :cotangentM, :coordinate, "dx")
#   _BASES[(vbundle, :frame)]      → Basis(:mf_M,  :tangentM,   :frame, "e")
#
# Convention for BasisElement indexing (ccf_M is the caller binding for M):
#   ccf_M[a1] → BasisElement(Basis(:ccf_M, :cotangentM, :coordinate, "dx"), ...)
#   cf_M[-a1] → BasisElement(Basis(:cf_M, :tangentM, :coordinate, "∂"), ...)
#
# Expansion Styles (New Architecture):
# basis_expansion now operates on a per-slot basis allowing mixed tensors.
#
#   basis_expansion(T)             → defaults to Coordinate style
#   basis_expansion(T, Coordinate) → Per-slot: uses :coordinate if available, falls back to :frame
#   basis_expansion(T, Frame)      → Per-slot: strictly uses :frame for all slots
#
# Depends on: indices.jl, manifolds.jl, tensorExpressions.jl
# =========================================

# =========================================
# 1.  Expansion Styles
# =========================================

abstract type ExpansionStyle end

struct CoordinateStyle <: ExpansionStyle end
struct FrameStyle <: ExpansionStyle end

const Coordinate = CoordinateStyle()
const Frame      = FrameStyle()

# =========================================
# 2.  Basis struct
# =========================================

"""
    Basis

A named frame for a vector bundle, of a given type (`:coordinate` or `:frame`).
Instances are created by [`@def_manifold`](@ref) (coordinate and frame) or
standalone [`@def_frame_bundle`](@ref) (frame only for custom bundles).

### Fields

- `name`     : binding symbol in the caller, e.g. `:cf_M`, `:ccf_M`
- `vbundle`  : the bundle this frame is for, e.g. `:cotangentM`
- `type`     : `:coordinate` (coordinate-induced) or `:frame` (arbitrary local frame)
- `print_as` : display label string, e.g. `"dx"`, `"∂"`, `"e"`, `"θ"`

Index with the **binding** (`ccf_M[a1]`); REPL compact `show` uses `print_as`:

    ccf_M[a1]    # a1 ∈ tangentM  → displays as dx[a1]
    mcf_M[-A1]   # -A1 ∈ cotangentM → displays as θ[A1]
"""
struct Basis
    name::Symbol      # binding :cf_M, :ccf_M, ...
    vbundle::Symbol   # :cotangentM, :tangentM, etc.
    type::Symbol      # :coordinate or :frame
    print_as::String  # display label: "dx", "∂", "e", "θ"
end

# =========================================
# 3.  BasisElement struct
# =========================================

"""
    BasisElement

A single element of a basis (coordinate or frame), constructed by `getindex`
on a [`Basis`](@ref).

### Fields

- `basis` : the [`Basis`](@ref) this element belongs to
- `index` : the [`AbstractIndex`](@ref) labeling this element;
            its vbundle is the **dual** of `basis.vbundle`

    ccf_M[a1] → BasisElement(Basis(:ccf_M, :cotangentM, :coordinate, "dx"), ...)
    mcf_M[A1] → BasisElement(Basis(:mcf_M, :cotangentM, :frame, "θ"), ...)
    cf_M[-a1] → BasisElement(Basis(:cf_M, :tangentM, :coordinate, "∂"), ...)
    mf_M[-A1] → BasisElement(Basis(:mf_M, :tangentM, :frame, "e"), ...)
"""
struct BasisElement
    basis::Basis
    index::AbstractIndex   # vbundle is the DUAL of basis.vbundle
end

# =========================================
# 4.  BasisExpansion struct
# =========================================

"""
    BasisExpansion

The formal basis expansion of a [`Tensor`](@ref) using canonical slot structures.

The `component` field is a [`TensorExpression`](@ref) built from the
tensor's canonical slot structure. The `basis_elements` give one
[`BasisElement`](@ref) per slot.

    T[-a1,-a2] dx[a1] ⊗ dx[a2]   # coordinate style
    T[-A1,-A2] θ[A1]  ⊗ θ[A2]    # frame style

Display rule: no `⊗` between `component` and the first basis element;
`⊗` only between consecutive basis elements.

### Fields

- `component`      : the [`TensorExpression`](@ref) giving the component part
- `basis_elements` : one [`BasisElement`](@ref) per slot
"""
struct BasisExpansion
    component::TensorExpression
    basis_elements::Vector{BasisElement}
end

# =========================================
# 5.  FrameBundle struct
# =========================================

"""
    FrameBundle

An arbitrary moving frame bundle associated with a vector bundle.
Created by [`@def_frame_bundle`](@ref) (standalone) or automatically by
[`@def_manifold`](@ref).

### Fields

- `name`    : symbol name, e.g. `:frameM`
- `vbundle` : the underlying vector bundle, e.g. `:tangentM`
- `dual`    : name of the dual frame bundle, e.g. `:coframeM`
- `basis`   : the frame [`Basis`](@ref) (type `:frame`) for this bundle

    frameM.basis    # → Basis(:mf_M, :tangentM, :frame, "e")
    coframeM.basis  # → Basis(:mcf_M, :cotangentM, :frame, "θ")
"""
struct FrameBundle
    name::Symbol       # :frameM
    vbundle::Symbol    # :tangentM  — the underlying VBundle
    dual::Symbol       # :coframeM
    basis::Basis       # Basis(:mf_M, :tangentM, :frame, "e")
end

# =========================================
# 6.  Module-level registries
# =========================================

"""
    _BASES :: Dict{Tuple{Symbol,Symbol}, Basis}

Maps `(vbundle_name, type)` to its [`Basis`](@ref).

    _BASES[(:cotangentM, :coordinate)] → Basis(:ccf_M, :cotangentM, :coordinate, "dx")
    _BASES[(:tangentM,   :coordinate)] → Basis(:cf_M,  :tangentM,   :coordinate, "∂")
    _BASES[(:cotangentM, :frame)]      → Basis(:mcf_M, :cotangentM, :frame, "θ")
    _BASES[(:tangentM,   :frame)]      → Basis(:mf_M,  :tangentM,   :frame, "e")

Populated by [`@def_manifold`](@ref) (both types) and standalone
[`@def_frame_bundle`](@ref) (frame only). Do not mutate directly.
"""
const _BASES = Dict{Tuple{Symbol,Symbol}, Basis}()

"""
    _BOUND_BASIS_SYMBOLS :: Dict{Symbol, Tuple{Symbol,Symbol,Symbol}}

Maps caller-scope **binding** symbols to `(vbundle, type, manifold)`.
Used to warn when a new [`@def_manifold`](@ref) would rebind an existing name.
"""
const _BOUND_BASIS_SYMBOLS = Dict{Symbol, Tuple{Symbol, Symbol, Symbol}}()

const _DEFAULT_PRINT_AS = ("∂", "dx", "e", "θ")
const _VALID_BASIS_TYPES = (:coordinate, :frame)

"""
    _FRAME_BUNDLES :: Dict{Symbol, FrameBundle}

Maps frame bundle names to their [`FrameBundle`](@ref) instances.

    _FRAME_BUNDLES[:frameM]   → FrameBundle(:frameM,   :tangentM,   :coframeM, ...)
    _FRAME_BUNDLES[:coframeM] → FrameBundle(:coframeM, :cotangentM, :frameM,   ...)

Populated by [`@def_manifold`](@ref) and [`@def_frame_bundle`](@ref).
"""
const _FRAME_BUNDLES = Dict{Symbol, FrameBundle}()

# =========================================
# 7.  Binding registry helpers
# =========================================

function _default_manifold_frame_bindings(m::Symbol)
    return (
        Symbol(:cf_, m),
        Symbol(:ccf_, m),
        Symbol(:mf_, m),
        Symbol(:mcf_, m),
    )
end

function _unregister_basis_bindings_for_vbundles!(vbundles::AbstractVector{Symbol})
    to_delete = Symbol[]
    for (sym, (vb, _typ, _m)) in _BOUND_BASIS_SYMBOLS
        vb in vbundles && push!(to_delete, sym)
    end
    for sym in to_delete
        delete!(_BOUND_BASIS_SYMBOLS, sym)
    end
    return nothing
end

function _warn_and_register_basis_binding!(
    binding::Symbol,
    vbundle::Symbol,
    btype::Symbol,
    manifold::Symbol,
)
    if haskey(_BOUND_BASIS_SYMBOLS, binding)
        old_vb, old_type, old_m = _BOUND_BASIS_SYMBOLS[binding]
        if old_vb != vbundle || old_type != btype
            @warn "Rebinding basis symbol :$binding (was $old_type basis on :$old_vb " *
                  "for manifold :$old_m; now $btype basis on :$vbundle for :$manifold). " *
                  "Use distinct names in frames=[...], e.g. frames=[cf_$manifold, ccf_$manifold, ...]."
        end
    end
    _BOUND_BASIS_SYMBOLS[binding] = (vbundle, btype, manifold)
    return nothing
end

# =========================================
# 8.  Equality and hashing
# =========================================

Base.:(==)(a::Basis, b::Basis) =
    a.name == b.name && a.vbundle == b.vbundle && a.type == b.type && a.print_as == b.print_as
Base.hash(b::Basis, h::UInt) = hash((b.name, b.vbundle, b.type, b.print_as), h)

Base.:(==)(a::BasisElement, b::BasisElement) = a.basis == b.basis && a.index == b.index
Base.hash(be::BasisElement, h::UInt) = hash((be.basis, be.index), h)

Base.:(==)(a::FrameBundle, b::FrameBundle) =
    a.name == b.name && a.vbundle == b.vbundle && a.dual == b.dual
Base.hash(fb::FrameBundle, h::UInt) = hash((fb.name, fb.vbundle, fb.dual), h)

# =========================================
# 8.  Stale-reference guard for FrameBundle
# =========================================

function Base.getproperty(fb::FrameBundle, field::Symbol)
    if !haskey(_FRAME_BUNDLES, getfield(fb, :name))
        @warn "FrameBundle :$(getfield(fb, :name)) has been undefined. " *
              "Variable still holds a stale reference."
        return nothing
    end
    getfield(fb, field)
end

function Base.propertynames(::FrameBundle, private::Bool=false)
    (:name, :vbundle, :dual, :basis)
end

# =========================================
# 9.  Registry accessors
# =========================================

"""
    basis_for_vbundle(vb::Symbol; type::Symbol=:coordinate) -> Basis

Return the [`Basis`](@ref) of the given `type` registered for vbundle `vb`.
Errors if no such basis has been registered.
"""
function basis_for_vbundle(vb::Symbol; type::Symbol=:coordinate)
    key = (vb, type)
    haskey(_BASES, key) ||
        error(
            "No $(type) basis registered for vbundle :$vb. " *
            "Call @def_manifold or @def_frame_bundle first."
        )
    _BASES[key]
end

"""
    bases_for_vbundle(vb::Symbol) -> Vector{Basis}

Return all [`Basis`](@ref) objects registered for vbundle `vb`, in order:
`:coordinate` first, then `:frame` (if present).

    bases_for_vbundle(:tangentM)
    # → [Basis(:cf_M, :tangentM, :coordinate, "∂"), Basis(:mf_M, :tangentM, :frame, "e")]
"""
function bases_for_vbundle(vb::Symbol)
    out = Basis[]
    for btype in _VALID_BASIS_TYPES
        key = (vb, btype)
        haskey(_BASES, key) && push!(out, _BASES[key])
    end
    return out
end

# =========================================
# 10. getindex — Basis[AbstractIndex] → BasisElement
# =========================================

"""
    Base.getindex(b::Basis, idx::AbstractIndex) -> BasisElement

Construct a [`BasisElement`](@ref) by applying basis `b` to index `idx`.
Validates:
1. `idx.vbundle` is the dual of `b.vbundle`
2. Index kind matches basis type: coordinate bases use [`CoordinateIndex`](@ref),
   frame bases use [`FrameIndex`](@ref)

# Examples
~~~julia
ccf_M[a1]   # a1 ∈ tangentM  — coordinate index on cotangent coordinate basis
cf_M[-a1]   # -a1 ∈ cotangentM — coordinate index on tangent coordinate basis
mf_M[-A1]   # -A1 ∈ cotangentM — basis index on tangent frame basis
mcf_M[A1]   # A1 ∈ tangentM  — basis index on cotangent frame basis
~~~
"""
function Base.getindex(b::Basis, idx::AbstractIndex)
    haskey(_VBUNDLES, b.vbundle) ||
        error("Basis references unregistered vbundle :$(b.vbundle).")
    dual_vb = _VBUNDLES[b.vbundle].dual
    idx.vbundle == dual_vb ||
        error(
            "BasisElement: index vbundle :$(idx.vbundle) is not the dual of " *
            "basis vbundle :$(b.vbundle). " *
            "Expected index from :$dual_vb. " *
            "Index the basis binding :$(b.name) (prints as $(b.print_as))."
        )

    if b.type === :coordinate
        idx isa CoordinateIndex ||
            error(
                "BasisElement: coordinate basis $(b.print_as) requires a " *
                "CoordinateIndex (e.g. a1 or -a1); got $(typeof(idx))."
            )
    elseif b.type === :frame
        idx isa FrameIndex ||
            error(
                "BasisElement: frame basis $(b.print_as) requires a " *
                "FrameIndex (e.g. A1 or -A1); got $(typeof(idx))."
            )
    end

    BasisElement(b, idx)
end

# =========================================
# 11. @def_frame_bundle macro
# =========================================
"""
    @def_frame_bundle frame_name vbundle_name basis_name cobasis_name

Create an arbitrary local frame bundle for `vbundle_name` and its dual.

- `frame_name`   : primal frame bundle name (e.g. `frameE`)
- `vbundle_name` : primal vbundle (must be in `_VBUNDLES`)
- `basis_name`   : frame basis for the primal bundle (e.g. `e`)
- `cobasis_name` : frame basis for the dual bundle (e.g. `θ`)

Dual frame bundle name: `Symbol("co", frame_name)` (e.g. `frameE` → `coframeE`).

Registers `_BASES[(vbundle, :frame)]` and `_FRAME_BUNDLES`.

# Example
~~~julia
@def_vbundle E M 3 [A1, A2, A3]
@def_frame_bundle frameE E eE θE
eE[-A1]   # BasisElement of E frame, labeled by dualE index
~~~
"""
macro def_frame_bundle(frame_name, vbundle_name, basis_name, cobasis_name)
    frame_name isa Symbol ||
        error("@def_frame_bundle: first argument must be a symbol, got $frame_name")
    vbundle_name isa Symbol ||
        error("@def_frame_bundle: second argument must be a symbol, got $vbundle_name")
    basis_name isa Symbol ||
        error("@def_frame_bundle: third argument must be a symbol, got $basis_name")
    cobasis_name isa Symbol ||
        error("@def_frame_bundle: fourth argument must be a symbol, got $cobasis_name")
    basis_name !== cobasis_name ||
        error(
            "@def_frame_bundle: basis_name and cobasis_name must be different, " *
            "both given as :$basis_name"
        )
    coframe_name   = Symbol("co", frame_name)
    fn_q           = QuoteNode(frame_name)
    cfn_q          = QuoteNode(coframe_name)
    vb_q           = QuoteNode(vbundle_name)
    basis_q        = QuoteNode(basis_name)
    cobasis_q      = QuoteNode(cobasis_name)
    primal_key_q   = QuoteNode((vbundle_name, :frame))

    quote
        haskey(_VBUNDLES, $(vb_q)) ||
            error(
                "@def_frame_bundle: vbundle $($(vb_q)) is not registered. " *
                "Call @def_manifold or @def_vbundle first."
            )

        local _fb_dual_vb     = _VBUNDLES[$(vb_q)].dual
        local _fb_dual_key    = (_fb_dual_vb, :frame)
        local _fb_manifold    = _VBUNDLES[$(vb_q)].manifold

        if haskey(_FRAME_BUNDLES, $(fn_q))
            @warn "Frame bundle $($(fn_q)) is already defined. Redefining."
        end

        _warn_and_register_basis_binding!($(basis_q),   $(vb_q),       :frame, _fb_manifold)
        _warn_and_register_basis_binding!($(cobasis_q), _fb_dual_vb,   :frame, _fb_manifold)

        _BASES[$(primal_key_q)] = Basis(
            $(basis_q), $(vb_q), :frame, $(QuoteNode(string(basis_name)))
        )
        _BASES[_fb_dual_key] = Basis(
            $(cobasis_q), _fb_dual_vb, :frame, $(QuoteNode(string(cobasis_name)))
        )

        _FRAME_BUNDLES[$(fn_q)]  = FrameBundle($(fn_q),  $(vb_q),    $(cfn_q), _BASES[$(primal_key_q)])
        _FRAME_BUNDLES[$(cfn_q)] = FrameBundle($(cfn_q), _fb_dual_vb, $(fn_q), _BASES[_fb_dual_key])

        $(esc(frame_name))    = _FRAME_BUNDLES[$(fn_q)]
        $(esc(coframe_name))  = _FRAME_BUNDLES[$(cfn_q)]
        $(esc(basis_name))    = _BASES[$(primal_key_q)]
        $(esc(cobasis_name))  = _BASES[_fb_dual_key]

        println(
            "Defined frame bundle $($(fn_q)) (frame basis $($(basis_q))) " *
            "and coframe bundle $($(cfn_q)) (frame cobasis $($(cobasis_q))) " *
            "over $(_fb_manifold)"
        )
        nothing
    end
end

# =========================================
# 12. @undef_frame_bundle macro
# =========================================
"""
@undef_frame_bundle frame_name
Remove the frame bundle frame_name, its dual coframe bundle, and both
:frame entries from _BASES. Silent if not registered.
"""
macro undef_frame_bundle(frame_name)
    frame_name isa Symbol ||
        error("@undef_frame_bundle: argument must be a symbol, got $frame_name")
    coframe_name = Symbol("co", frame_name)
    fn_q  = QuoteNode(frame_name)
    cfn_q = QuoteNode(coframe_name)
    quote
        if haskey(_FRAME_BUNDLES, $(fn_q))
            local _ufb_vb      = getfield(_FRAME_BUNDLES[$(fn_q)], :vbundle)
            local _ufb_dvb     = _VBUNDLES[_ufb_vb].dual
            _unregister_basis_bindings_for_vbundles!([_ufb_vb, _ufb_dvb])
            delete!(_BASES, (_ufb_vb,  :frame))
            delete!(_BASES, (_ufb_dvb, :frame))
            delete!(_FRAME_BUNDLES, $(fn_q))
            delete!(_FRAME_BUNDLES, $(cfn_q))
        end
        nothing
    end
end

# =========================================
# 13. Per-slot Basis resolution
# =========================================
function _resolve_basis_type(slot_vb::Symbol, ::CoordinateStyle)
    haskey(_BASES, (slot_vb, :coordinate)) && return :coordinate
    haskey(_BASES, (slot_vb, :frame))      && return :frame
    error("No basis registered for vbundle :$slot_vb.")
end

function _resolve_basis_type(slot_vb::Symbol, ::FrameStyle)
    haskey(_BASES, (slot_vb, :frame)) && return :frame
    error("No frame basis registered for vbundle :$slot_vb.")
end

# =========================================
# 14. Canonical Indices & Component Generation
# =========================================
function _canonical_indices(T::Tensor, style::ExpansionStyle)::TensorExpression
    haskey(_MANIFOLDS, T.manifold) ||
        error("basis_expansion: tensor references unregistered manifold :$(T.manifold).")
    n = T.rank
    canonical_idxs = AbstractIndex[]
    # We track how many index symbols have been consumed for each bundle pool.
    # A bundle and its dual share the same pool of registered indices.
    usage_counts = Dict{Tuple{Symbol, Symbol}, Int}() 

    for i in 1:n
        slot_vb = T.slots[i]
        btype = _resolve_basis_type(slot_vb, style)

        index_registry = btype === :coordinate ? _COORDINATE_INDICES : _FRAME_INDICES
        index_constructor = btype === :coordinate ? CoordinateIndex : FrameIndex

        dual_vb = _VBUNDLES[slot_vb].dual
        pool_id = min(slot_vb, dual_vb) # Unique ID for the primal/dual pair
        
        syms = sort([s for (s, home) in index_registry if home == slot_vb || home == dual_vb])

        count = get(usage_counts, (pool_id, btype), 0) + 1
        usage_counts[(pool_id, btype)] = count

        if count > length(syms)
            error(
                "basis_expansion: tensor exceeds number of registered $btype indices " *
                "for bundles :$slot_vb / :$dual_vb. Add more with @add_indices."
            )
        end

        push!(canonical_idxs, index_constructor(syms[count], slot_vb))
    end

    TensorExpression(T, canonical_idxs)
end

function _basis_element_for_slot(
    slot_vb  :: Symbol,
    cat      :: Symbol,
    idx      :: AbstractIndex,
)::BasisElement
    haskey(_BASES, (slot_vb, cat)) ||
        error("basis_expansion: no $(cat) basis registered for vbundle :$slot_vb.")
    b = _BASES[(slot_vb, cat)]
    dual_vb = _VBUNDLES[slot_vb].dual
    if idx isa CoordinateIndex
        dual_idx = CoordinateIndex(idx.symbol, dual_vb)
    elseif idx isa FrameIndex
        dual_idx = FrameIndex(idx.symbol, dual_vb)
    else
        error("basis_expansion: unsupported index type $(typeof(idx))")
    end

    BasisElement(b, dual_idx)
end

# =========================================
# 15. basis_expansion — public API
# =========================================
"""
    basis_expansion(T::Tensor, style::ExpansionStyle=Coordinate) -> BasisExpansion
    basis_expansion(T::Tensor) -> BasisExpansion

Expand tensor `T` slot-by-slot using canonical indices and the given [`ExpansionStyle`](@ref).

- **`Coordinate`** (default): per slot use `:coordinate` basis if registered, else `:frame`.
- **`Frame`**: per slot use `:frame` basis only.

# Examples
~~~julia
basis_expansion(T)             # Coordinate style
basis_expansion(T, Coordinate)
basis_expansion(T, Frame)
~~~
"""
basis_expansion(T::Tensor) = basis_expansion(T, Coordinate)

function basis_expansion(T::Tensor, style::ExpansionStyle)::BasisExpansion
    comp = _canonical_indices(T, style)
    basis_elements = BasisElement[]
    for i in 1:T.rank
        slot_vb = T.slots[i]
        btype = _resolve_basis_type(slot_vb, style)
        idx = comp.indices[i]
        push!(basis_elements, _basis_element_for_slot(slot_vb, btype, idx))
    end
    BasisExpansion(comp, basis_elements)
end

# =========================================
# 16. show — Basis
# =========================================
function Base.show(io::IO, ::MIME"text/plain", b::Basis)
    print(io, b.print_as)
end

function Base.show(io::IO, ::MIME"text/html", b::Basis)
    print(io, b.print_as)
end

# =========================================
# 17. show — FrameBundle
# =========================================
function Base.show(io::IO, ::MIME"text/plain", fb::FrameBundle)
    print(io, "FrameBundle($(fb.name), vbundle=$(fb.vbundle), dual=$(fb.dual), basis=$(fb.basis.print_as))")
end

function Base.show(io::IO, ::MIME"text/html", fb::FrameBundle)
    print(io, """
    <div style="border:1px solid #ddd;padding:10px;border-radius:5px;background:#f4faff;">
        <h4 style="margin-top:0;">FrameBundle: <span style="color:#0d6efd;">$(fb.name)</span></h4>
        <table style="width:100%;border-collapse:collapse;">
            <tr><td style="font-weight:bold;width:150px;text-align:left;">VBundle</td><td><code>$(fb.vbundle)</code></td></tr>
            <tr><td style="font-weight:bold;text-align:left;">Dual</td><td><code>$(fb.dual)</code></td></tr>
            <tr><td style="font-weight:bold;text-align:left;">Frame basis</td><td><code>$(fb.basis.print_as)</code></td></tr>
        </table>
    </div>
    """)
end

# =========================================
# 18. show — BasisElement
# =========================================

function Base.show(io::IO, ::MIME"text/plain", be::BasisElement)
    idx    = be.index
    prefix = (haskey(_VBUNDLES, idx.vbundle) && is_down(idx)) ? "-" : ""
    print(io, "$(be.basis.print_as)[$(prefix)$(idx.symbol)]")
end

# ── LaTeX helpers ────────────────────────────────────────────────────────────
function _basis_latex_name(label::String)::String
    if label == "dx"
        return "\\mathrm{d}x"
    elseif label == "∂"
        return "\\partial"
    elseif label == "θ"
        return "\\theta"
    elseif label == "e"
        return "e"
    else
        return "\\mathrm{$(label)}"
    end
end

function _basis_latex_index(sym::Symbol)::String
    s = string(sym)
    m = match(r"^([^\d]*)(\d+)$", s)
    m === nothing ? s : "$(m[1])_{$(m[2])}"
end

function _format_basis_element_latex(be::BasisElement)::String
    name_str = _basis_latex_name(be.basis.print_as)
    idx_str  = _basis_latex_index(be.index.symbol)
    if haskey(_VBUNDLES, be.index.vbundle) && is_down(be.index)
        return "$(name_str)_{$(idx_str)}"
    else
        return "$(name_str)^{$(idx_str)}"
    end
end

function Base.show(io::IO, ::MIME"text/latex", be::BasisElement)
    print(io, "\$", _format_basis_element_latex(be), "\$")
end

# ── HTML helpers ─────────────────────────────────────────────────────────────
function _format_basis_element_html(be::BasisElement)::String
    name_str = be.basis.print_as
    idx_str  = string(be.index.symbol)
    if haskey(_VBUNDLES, be.index.vbundle) && is_down(be.index)
        return "$(name_str)<sub>$(idx_str)</sub>"
    else
        return "$(name_str)<sup>$(idx_str)</sup>"
    end
end

function Base.show(io::IO, ::MIME"text/html", be::BasisElement)
    print(io, _format_basis_element_html(be))
end

# =========================================
# 19. show — BasisExpansion
# =========================================
"""
Base.show(io::IO, bx::BasisExpansion)
REPL display: component followed by basis elements joined with ⊗.
No ⊗ between the component and the first basis element.
T[-a1,-a2] dx[a1] ⊗ dx[a2]
T[-A1,-A2] θ[A1]  ⊗ θ[A2]
S[a1,-a2]  ∂[-a1] ⊗ dx[a2]
"""
function Base.show(io::IO, ::MIME"text/plain", bx::BasisExpansion)
    print(io, bx.component)
    if !isempty(bx.basis_elements)
        print(io, " ")
        print(io, join(string.(bx.basis_elements), " ⊗ "))
    end
end

"""
Base.show(io::IO, ::MIME"text/latex", bx::BasisExpansion)
LaTeX display:
T_{a_{1} a_{2}}\\, \\mathrm{d}x^{a_{1}} \\otimes \\mathrm{d}x^{a_{2}}
"""
function Base.show(io::IO, ::MIME"text/latex", bx::BasisExpansion)
    comp_str  = sprint(show, MIME"text/latex"(), bx.component)
    comp_core = strip(comp_str, ['\$'])
    basis_strs = [_format_basis_element_latex(be) for be in bx.basis_elements]
    result = comp_core
    if !isempty(basis_strs)
        result *= "\\," * join(basis_strs, " \\otimes ")
    end
    print(io, "\$", result, "\$")
end

"""
Base.show(io::IO, ::MIME"text/html", bx::BasisExpansion)
HTML display using  /  tags.
The component uses _format_html from tensorExpressions.jl.
"""
function Base.show(io::IO, ::MIME"text/html", bx::BasisExpansion)
    comp_html = _format_html(bx.component)
    if isempty(bx.basis_elements)
        print(io, comp_html)
        return
    end
    basis_html = join(
        [_format_basis_element_html(be) for be in bx.basis_elements],
        " ⊗ "
    )
    print(io, comp_html, " ", basis_html)
end

# =========================================
# Exports
# =========================================
export Basis, BasisElement, BasisExpansion, FrameBundle
export ExpansionStyle, CoordinateStyle, FrameStyle, Coordinate, Frame
export _BASES, _FRAME_BUNDLES, _BOUND_BASIS_SYMBOLS
export basis_for_vbundle, bases_for_vbundle
export @def_frame_bundle, @undef_frame_bundle
export basis_expansion