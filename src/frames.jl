# =========================================
# frames.jl — SymbolicTensors.jl
#
# Frame bundles: coordinate frames (∂/dx) and arbitrary local frames (e/θ).
#
# Each vbundle may carry two basis categories:
#
#   :coordinate — coordinate-induced frame/coframe when geometrically available
#                 registered by @def_manifold as part of vbundle creation
#   :frame      — arbitrary chosen local frame/coframe
#                 registered by @def_frame_bundle and bound as FrameBundle variables
#
# Registry:
#   _BASES[(vbundle, :coordinate)] → Basis(:∂,  :tangentM,   :coordinate)
#   _BASES[(vbundle, :frame)]      → Basis(:e,  :tangentM,   :frame)
#
# Convention for BasisElement indexing:
#   dx[a1]   → BasisElement(Basis(:dx,:cotangentM,:coordinate), CoordinateIndex(:a1,:tangentM))
#   ∂[-a1]   → BasisElement(Basis(:∂, :tangentM,  :coordinate), CoordinateIndex(:a1,:cotangentM))
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

A named frame for a vector bundle, of a given category (`:coordinate` or `:frame`).
Instances are created by [`@def_manifold`](@ref) (coordinate and frame) or
standalone [`@def_frame_bundle`](@ref) (frame only for custom bundles).

### Fields

- `name`     : display name, e.g. `:dx`, `:∂`, `:e`, `:θ`
- `vbundle`  : the bundle this frame is for, e.g. `:cotangentM`
- `category` : `:coordinate` (coordinate-induced) or `:frame` (arbitrary chosen local frame)

Indexing a `Basis` with an [`AbstractIndex`](@ref) from the **dual** bundle
produces a [`BasisElement`](@ref):

    dx[a1]    # a1 ∈ tangentM  → BasisElement of cotangentM coordinate frame
    e[-a1]    # -a1 ∈ cotangentM → BasisElement of tangentM arbitrary frame
"""
struct Basis
    name::Symbol      # display name: :dx, :∂, :e, :θ, or user-defined
    vbundle::Symbol   # bundle this frame is for: :cotangentM, :tangentM, etc.
    category::Symbol  # :coordinate or :frame
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

    dx[a1]   → BasisElement(Basis(:dx, :cotangentM, :coordinate), CoordinateIndex(:a1, :tangentM))
    θ[A1]    → BasisElement(Basis(:θ,  :cotangentM, :frame),      BasisIndex(:A1, :tangentM))
    ∂[-a1]   → BasisElement(Basis(:∂,  :tangentM,   :coordinate), CoordinateIndex(:a1, :cotangentM))
    e[-A1]   → BasisElement(Basis(:e,  :tangentM,   :frame),      BasisIndex(:A1, :cotangentM))
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
- `basis`   : the frame [`Basis`](@ref) (category `:frame`) for this bundle

    frameM.basis    # → Basis(:e, :tangentM, :frame)
    coframeM.basis  # → Basis(:θ, :cotangentM, :frame)
"""
struct FrameBundle
    name::Symbol       # :frameM
    vbundle::Symbol    # :tangentM  — the underlying VBundle
    dual::Symbol       # :coframeM
    basis::Basis       # Basis(:e, :tangentM, :frame)
end

# =========================================
# 6.  Module-level registries
# =========================================

"""
    _BASES :: Dict{Tuple{Symbol,Symbol}, Basis}

Maps `(vbundle_name, category)` to its [`Basis`](@ref).

    _BASES[(:cotangentM, :coordinate)] → Basis(:dx, :cotangentM, :coordinate)
    _BASES[(:tangentM,   :coordinate)] → Basis(:∂,  :tangentM,   :coordinate)
    _BASES[(:cotangentM, :frame)]      → Basis(:θ,  :cotangentM, :frame)
    _BASES[(:tangentM,   :frame)]      → Basis(:e,  :tangentM,   :frame)

Populated by [`@def_manifold`](@ref) (both categories) and standalone
[`@def_frame_bundle`](@ref) (frame only). Do not mutate directly.
"""
const _BASES = Dict{Tuple{Symbol,Symbol}, Basis}()

"""
    _FRAME_BUNDLES :: Dict{Symbol, FrameBundle}

Maps frame bundle names to their [`FrameBundle`](@ref) instances.

    _FRAME_BUNDLES[:frameM]   → FrameBundle(:frameM,   :tangentM,   :coframeM, ...)
    _FRAME_BUNDLES[:coframeM] → FrameBundle(:coframeM, :cotangentM, :frameM,   ...)

Populated by [`@def_manifold`](@ref) and [`@def_frame_bundle`](@ref).
"""
const _FRAME_BUNDLES = Dict{Symbol, FrameBundle}()

# The two valid basis categories.
const _VALID_BASIS_CATEGORIES = (:coordinate, :frame)

# =========================================
# 7.  Equality and hashing
# =========================================

Base.:(==)(a::Basis, b::Basis) =
    a.name == b.name && a.vbundle == b.vbundle && a.category == b.category
Base.hash(b::Basis, h::UInt) = hash((b.name, b.vbundle, b.category), h)

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
    basis_for_vbundle(vb::Symbol; category::Symbol=:coordinate) -> Basis

Return the [`Basis`](@ref) of the given `category` registered for vbundle `vb`.
Errors if no such basis has been registered.
"""
function basis_for_vbundle(vb::Symbol; category::Symbol=:coordinate)
    key = (vb, category)
    haskey(_BASES, key) ||
        error(
            "No $(category) basis registered for vbundle :$vb. " *
            "Call @def_manifold or @def_frame_bundle first."
        )
    _BASES[key]
end

"""
    bases_for_vbundle(vb::Symbol) -> Vector{Basis}

Return all [`Basis`](@ref) objects registered for vbundle `vb`, in order:
`:coordinate` first, then `:frame` (if present).

    bases_for_vbundle(:tangentM)
    # → [Basis(:∂, :tangentM, :coordinate), Basis(:e, :tangentM, :frame)]
"""
function bases_for_vbundle(vb::Symbol)
    out = Basis[]
    for cat in _VALID_BASIS_CATEGORIES
        key = (vb, cat)
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
2. Index kind matches basis category: coordinate bases use [`CoordinateIndex`](@ref),
   frame bases use [`BasisIndex`](@ref)

# Examples
~~~julia
dx[a1]      # a1 ∈ tangentM  — coordinate index on cotangent coordinate basis
∂[-a1]      # -a1 ∈ cotangentM — coordinate index on tangent coordinate basis
e[-A1]      # -A1 ∈ cotangentM — basis index on tangent frame basis
θ[A1]       # A1 ∈ tangentM  — basis index on cotangent frame basis
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
            "Use $(b.name)[...] with an index from the dual bundle."
        )
        
    if b.category === :coordinate
        idx isa CoordinateIndex ||
            error(
                "BasisElement: coordinate basis :$(b.name) requires a " *
                "CoordinateIndex (e.g. a1 or -a1); got $(typeof(idx))."
            )
    elseif b.category === :frame
        idx isa BasisIndex ||
            error(
                "BasisElement: frame basis :$(b.name) requires a " *
                "BasisIndex (e.g. A1 or -A1); got $(typeof(idx))."
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

        _BASES[$(primal_key_q)] = Basis($(basis_q),   $(vb_q),    :frame)
        _BASES[_fb_dual_key]    = Basis($(cobasis_q), _fb_dual_vb, :frame)

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
function _resolve_basis_category(slot_vb::Symbol, ::CoordinateStyle)
    haskey(_BASES, (slot_vb, :coordinate)) && return :coordinate
    haskey(_BASES, (slot_vb, :frame))      && return :frame
    error("No basis registered for vbundle :$slot_vb.")
end

function _resolve_basis_category(slot_vb::Symbol, ::FrameStyle)
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
        cat = _resolve_basis_category(slot_vb, style)

        index_registry = cat === :coordinate ? _COORDINATE_INDICES : _BASIS_INDICES
        index_constructor = cat === :coordinate ? CoordinateIndex : BasisIndex

        dual_vb = _VBUNDLES[slot_vb].dual
        pool_id = min(slot_vb, dual_vb) # Unique ID for the primal/dual pair
        
        syms = sort([s for (s, home) in index_registry if home == slot_vb || home == dual_vb])

        count = get(usage_counts, (pool_id, cat), 0) + 1
        usage_counts[(pool_id, cat)] = count

        if count > length(syms)
            error(
                "basis_expansion: tensor exceeds number of registered $cat indices " *
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
    elseif idx isa BasisIndex
        dual_idx = BasisIndex(idx.symbol, dual_vb)
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

Handles mixed-variance and multi-bundle tensors. There is no `basis_expansion(T, ::Basis)` overload.

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
        cat = _resolve_basis_category(slot_vb, style)
        idx = comp.indices[i]
        push!(basis_elements, _basis_element_for_slot(slot_vb, cat, idx))
    end
    BasisExpansion(comp, basis_elements)
end

# =========================================
# 16. show — Basis
# =========================================
function Base.show(io::IO, b::Basis)
    cat_label = b.category === :coordinate ? "coord" : "frame"
    print(io, "Basis($(b.name), $(b.vbundle), $(cat_label))")
end

# =========================================
# 17. show — FrameBundle
# =========================================
function Base.show(io::IO, fb::FrameBundle)
    print(io, "FrameBundle($(fb.name), vbundle=$(fb.vbundle), dual=$(fb.dual), basis=$(fb.basis.name))")
end

function Base.show(io::IO, ::MIME"text/html", fb::FrameBundle)
    print(io, """
    <div style="border:1px solid #ddd;padding:10px;border-radius:5px;background:#f4faff;">
        <h4 style="margin-top:0;">FrameBundle: <span style="color:#0d6efd;">$(fb.name)</span></h4>
        <table style="width:100%;border-collapse:collapse;">
            <tr><td style="font-weight:bold;width:150px;text-align:left;">VBundle</td><td><code>$(fb.vbundle)</code></td></tr>
            <tr><td style="font-weight:bold;text-align:left;">Dual</td><td><code>$(fb.dual)</code></td></tr>
            <tr><td style="font-weight:bold;text-align:left;">Frame basis</td><td><code>$(fb.basis.name)</code></td></tr>
        </table>
    </div>
    """)
end

# =========================================
# 18. show — BasisElement
# =========================================
function Base.show(io::IO, be::BasisElement)
    idx    = be.index
    prefix = (haskey(_VBUNDLES, idx.vbundle) && is_down(idx)) ? "-" : ""
    print(io, "$(be.basis.name)[$(prefix)$(idx.symbol)]")
end

# ── LaTeX helpers ────────────────────────────────────────────────────────────
function _basis_latex_name(sym::Symbol)::String
    if sym === :dx
        return "\\mathrm{d}x"
    elseif sym === :∂
        return "\\partial"
    elseif sym === :θ
        return "\\theta"
    elseif sym === :e
        return "e"
    else
        return "\\mathrm{$(string(sym))}"
    end
end

function _basis_latex_index(sym::Symbol)::String
    s = string(sym)
    m = match(r"^([^\d]*)(\d+)$", s)
    m === nothing ? s : "$(m[1])_{$(m[2])}"
end

function _format_basis_element_latex(be::BasisElement)::String
    name_str = _basis_latex_name(be.basis.name)
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
    name_str = string(be.basis.name)
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
function Base.show(io::IO, bx::BasisExpansion)
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
export _BASES, _FRAME_BUNDLES
export basis_for_vbundle, bases_for_vbundle
export @def_frame_bundle, @undef_frame_bundle
export basis_expansion