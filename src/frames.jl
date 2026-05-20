# =========================================
# frames.jl — AbstractTensors.jl
#
# Frame bundles: coordinate frames (∂/dx) and moving frames (e/θ).
#
# Each manifold carries two frame types for its tangent/cotangent bundles:
#
#   :coordinate  — the natural coordinate frame (∂ for tangentM, dx for cotangentM)
#                  registered by @def_manifold as part of vbundle creation
#   :moving      — a user-defined moving frame (e/θ by default)
#                  registered by @def_frame_bundle and bound as FrameBundle variables
#
# Registry:
#   _BASES[(vbundle, :coordinate)] → Basis(:∂,  :tangentM,   :coordinate)
#   _BASES[(vbundle, :moving)]     → Basis(:e,   :tangentM,   :moving)
#
# Convention for BasisElement indexing (unchanged):
#   dx[a1]   → BasisElement(Basis(:dx,:cotangentM,:coordinate), CoordinateIndex(:a1,:tangentM))
#   ∂[-a1]   → BasisElement(Basis(:∂, :tangentM,  :coordinate), CoordinateIndex(:a1,:cotangentM))
#
# basis_expansion(T[-a1,-a2])               →  T[-a1,-a2] dx[a1] ⊗ dx[a2]
# basis_expansion(T[-a1,-a2]; frame=:moving) →  T[-a1,-a2] θ[a1]  ⊗ θ[a2]
#
# Macro hierarchy:
#   @def_manifold  ── inline coord frame registration  (binds ∂, dx)
#                  └─ inline moving frame registration (binds e, θ, frameM, coframeM)
#   @def_frame_bundle ── standalone primitive for custom vbundles
#
# Depends on: indices.jl, manifolds.jl, tensorExpressions.jl
# =========================================


# =========================================
# 1.  Basis struct
# =========================================

"""
    Basis

A named frame for a vector bundle, of a given type (`:coordinate` or `:moving`).
Instances are normally obtained from [`@def_manifold`](@ref) (coordinate and moving)
or [`@def_frame_bundle`](@ref) (moving only on custom bundles), which bind variables
such as `dx`, `∂`, `e`, and `θ` in the caller's scope.

### Fields

- `name`    : display name, e.g. `:dx`, `:∂`, `:e`, `:θ`
- `vbundle` : the bundle this frame is for, e.g. `:cotangentM`
- `type`    : `:coordinate` (natural frame) or `:moving` (user-defined frame)

### Construction

The struct constructor takes three **symbols** (note the leading `:`):

```julia
Basis(:dx, :cotangentM, :coordinate)
Basis(:∂,  :tangentM,   :coordinate)
Basis(:e,  :tangentM,   :moving)
```

Lookup without the bound variable:

```julia
basis_for_vbundle(:cotangentM; type=:coordinate)  # same object as dx
bases_for_vbundle(:tangentM)                       # coordinate + moving bases
```

### Indexing

Indexing a `Basis` with an [`AbstractIndex`](@ref) from the **dual** bundle
produces a [`BasisElement`](@ref):

```julia
dx[a1]     # a1 ∈ tangentM   → cotangentM coordinate element
e[-A1]     # -A1 ∈ cotangentM → tangentM moving element
```
"""
struct Basis
    name::Symbol      # display name: :dx, :∂, :e, :θ, or user-defined
    vbundle::Symbol   # bundle this frame is for: :cotangentM, :tangentM, etc.
    type::Symbol      # :coordinate  or  :moving
end


# =========================================
# 2.  BasisElement struct
# =========================================

"""
    BasisElement

A single element of a frame (coordinate or moving), constructed by `getindex`
on a [`Basis`](@ref).

### Fields

- `basis` : the [`Basis`](@ref) this element belongs to
- `index` : the [`AbstractIndex`](@ref) labeling this element;
            its vbundle is the **dual** of `basis.vbundle`

    dx[a1]   → BasisElement(Basis(:dx, :cotangentM, :coordinate), CoordinateIndex(:a1, :tangentM))
    θ[A1]    → BasisElement(Basis(:θ,  :cotangentM, :moving),     BasisIndex(:A1, :tangentM))
    ∂[-a1]   → BasisElement(Basis(:∂,  :tangentM,   :coordinate), CoordinateIndex(:a1, :cotangentM))
    e[-A1]   → BasisElement(Basis(:e,  :tangentM,   :moving),     BasisIndex(:A1, :cotangentM))
"""
struct BasisElement
    basis::Basis
    index::AbstractIndex   # vbundle is the DUAL of basis.vbundle
end


# =========================================
# 3.  BasisExpansion struct
# =========================================

"""
    BasisExpansion

The formal basis expansion of a [`TensorExpression`](@ref) in either the
coordinate or moving frame.

    T[-a1,-a2] dx[a1] ⊗ dx[a2]   # coordinate frame (default)
    T[-a1,-a2] θ[a1]  ⊗ θ[a2]    # moving frame

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
# 4.  FrameBundle struct
# =========================================

"""
    FrameBundle

A moving frame bundle associated with a vector bundle.
Created by [`@def_frame_bundle`](@ref) (standalone) or automatically by
[`@def_manifold`](@ref).

### Fields

- `name`    : symbol name, e.g. `:frameM`
- `vbundle` : the underlying vector bundle, e.g. `:tangentM`
- `dual`    : name of the dual frame bundle, e.g. `:coframeM`
- `basis`   : the moving [`Basis`](@ref) (type `:moving`) for this bundle

    frameM.basis    # → Basis(:e, :tangentM, :moving)
    coframeM.basis  # → Basis(:θ, :cotangentM, :moving)
"""
struct FrameBundle
    name::Symbol       # :frameM
    vbundle::Symbol    # :tangentM  — the underlying VBundle
    dual::Symbol       # :coframeM
    basis::Basis       # Basis(:e, :tangentM, :moving)
end


# =========================================
# 5.  Module-level registries
# =========================================

"""
    _BASES :: Dict{Tuple{Symbol,Symbol}, Basis}

Maps `(vbundle_name, frame_type)` to its [`Basis`](@ref).

    _BASES[(:cotangentM, :coordinate)] → Basis(:dx, :cotangentM, :coordinate)
    _BASES[(:tangentM,   :coordinate)] → Basis(:∂,  :tangentM,   :coordinate)
    _BASES[(:cotangentM, :moving)]     → Basis(:θ,  :cotangentM, :moving)
    _BASES[(:tangentM,   :moving)]     → Basis(:e,  :tangentM,   :moving)

Populated by [`@def_manifold`](@ref) (both types) and standalone
[`@def_frame_bundle`](@ref) (moving only). Do not mutate directly.
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


# =========================================
# 6.  Equality and hashing
# =========================================

Base.:(==)(a::Basis, b::Basis) =
    a.name == b.name && a.vbundle == b.vbundle && a.type == b.type
Base.hash(b::Basis, h::UInt) = hash((b.name, b.vbundle, b.type), h)

Base.:(==)(a::BasisElement, b::BasisElement) = a.basis == b.basis && a.index == b.index
Base.hash(be::BasisElement, h::UInt) = hash((be.basis, be.index), h)

Base.:(==)(a::FrameBundle, b::FrameBundle) =
    a.name == b.name && a.vbundle == b.vbundle && a.dual == b.dual
Base.hash(fb::FrameBundle, h::UInt) = hash((fb.name, fb.vbundle, fb.dual), h)


# =========================================
# 7.  Stale-reference guard for FrameBundle
# =========================================

"""
    Base.getproperty(fb::FrameBundle, field::Symbol)

Field access for `FrameBundle` instances.
Checks that `fb` is still registered in `_FRAME_BUNDLES` before returning the
requested field, turning stale references into an immediate warning.
"""
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
# 8.  Registry accessors
# =========================================

"""
    basis_for_vbundle(vb::Symbol; type::Symbol=:coordinate) -> Basis

Return the [`Basis`](@ref) of the given `type` registered for vbundle `vb`.
Errors if no such frame has been registered.
"""
function basis_for_vbundle(vb::Symbol; type::Symbol=:coordinate)
    key = (vb, type)
    haskey(_BASES, key) ||
        error(
            "No $(type) frame registered for vbundle :$vb. " *
            "Call @def_manifold or @def_frame_bundle first."
        )
    _BASES[key]
end

"""
    bases_for_vbundle(vb::Symbol) -> Vector{Basis}

Return all [`Basis`](@ref) objects registered for vbundle `vb`, in order:
`:coordinate` first, then `:moving` (if present).

    bases_for_vbundle(:tangentM)
    # → [Basis(:∂, :tangentM, :coordinate), Basis(:e, :tangentM, :moving)]
"""
function bases_for_vbundle(vb::Symbol)
    out = Basis[]
    for frame_type in (:coordinate, :moving)
        key = (vb, frame_type)
        haskey(_BASES, key) && push!(out, _BASES[key])
    end
    return out
end


# =========================================
# 9.  getindex — Basis[AbstractIndex] → BasisElement
# =========================================

"""
    Base.getindex(b::Basis, idx::AbstractIndex) -> BasisElement

Construct a [`BasisElement`](@ref) by applying basis `b` to index `idx`.

Validates that `idx.vbundle` is the dual of `b.vbundle`:

    dx[a1]    # a1 ∈ tangentM  → valid for cotangentM basis (coordinate)
    θ[a1]     # a1 ∈ tangentM  → valid for cotangentM basis (moving)
    ∂[-a1]    # -a1 ∈ cotangentM → valid for tangentM basis (coordinate)
    e[-a1]    # -a1 ∈ cotangentM → valid for tangentM basis (moving)
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
    BasisElement(b, idx)
end


# =========================================
# 10.  @def_frame_bundle macro
# =========================================

"""
    @def_frame_bundle frame_name vbundle_name basis_name cobasis_name

Create a moving frame bundle for `vbundle_name` and its dual bundle.

- `frame_name`    : name for the primal frame bundle (e.g. `frameE`)
- `vbundle_name`  : the primal vbundle (must already be in `_VBUNDLES`)
- `basis_name`    : moving basis name for the primal bundle (e.g. `e`)
- `cobasis_name`  : moving basis name for the dual bundle (e.g. `θ`)

The dual frame bundle name is derived as `Symbol("co", frame_name)`
(e.g. `frameE` → `coframeE`).

Registers:
- `_BASES[(vbundle_name, :moving)]` and `_BASES[(dual_vbundle, :moving)]`
- `_FRAME_BUNDLES[frame_name]` and `_FRAME_BUNDLES[coframe_name]`

Binds `frame_name`, `coframe_name`, `basis_name`, and `cobasis_name`
as variables in the caller's scope.

The two basis names must differ.

# Examples
```julia
@def_manifold M 4 [a1, a2, a3, a4] [A1, A2, A3, A4]   # frameM/coframeM already created with e/θ

# Standalone use for a custom vbundle:
@def_vbundle E M 3 [A1, A2, A3]
@def_frame_bundle frameE E eE θE
# frameE and coframeE bound as FrameBundle; eE and θE bound as Basis (moving)
eE[-A1]   # BasisElement of E moving frame, labeled by dualE index
```
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
    primal_key_q   = QuoteNode((vbundle_name, :moving))
    # dual key computed at runtime; use a gensym for hygiene

    quote
        haskey(_VBUNDLES, $(vb_q)) ||
            error(
                "@def_frame_bundle: vbundle $($(vb_q)) is not registered. " *
                "Call @def_manifold or @def_vbundle first."
            )

        local _fb_dual_vb     = _VBUNDLES[$(vb_q)].dual
        local _fb_dual_key    = (_fb_dual_vb, :moving)
        local _fb_manifold    = _VBUNDLES[$(vb_q)].manifold

        if haskey(_FRAME_BUNDLES, $(fn_q))
            @warn "Frame bundle $($(fn_q)) is already defined. Redefining."
        end

        _BASES[$(primal_key_q)] = Basis($(basis_q),   $(vb_q),    :moving)
        _BASES[_fb_dual_key]    = Basis($(cobasis_q), _fb_dual_vb, :moving)

        _FRAME_BUNDLES[$(fn_q)]  = FrameBundle($(fn_q),  $(vb_q),    $(cfn_q), _BASES[$(primal_key_q)])
        _FRAME_BUNDLES[$(cfn_q)] = FrameBundle($(cfn_q), _fb_dual_vb, $(fn_q), _BASES[_fb_dual_key])

        $(esc(frame_name))    = _FRAME_BUNDLES[$(fn_q)]
        $(esc(coframe_name))  = _FRAME_BUNDLES[$(cfn_q)]
        $(esc(basis_name))    = _BASES[$(primal_key_q)]
        $(esc(cobasis_name))  = _BASES[_fb_dual_key]

        println(
            "Defined frame bundle $($(fn_q)) (moving frame $($(basis_q))) " *
            "and coframe bundle $($(cfn_q)) (moving coframe $($(cobasis_q))) " *
            "over $(_fb_manifold)"
        )
        nothing
    end
end


# =========================================
# 11.  @undef_frame_bundle macro
# =========================================

"""
    @undef_frame_bundle frame_name

Remove the frame bundle `frame_name`, its dual coframe bundle, and both
`:moving` entries from `_BASES`. Silent if not registered.
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
            delete!(_BASES, (_ufb_vb,  :moving))
            delete!(_BASES, (_ufb_dvb, :moving))
            delete!(_FRAME_BUNDLES, $(fn_q))
            delete!(_FRAME_BUNDLES, $(cfn_q))
        end
        nothing
    end
end


# =========================================
# 12.  basis_expansion
# =========================================

"""
    basis_expansion(te::TensorExpression; frame::Symbol=:coordinate) -> BasisExpansion

Expand `te` into its formal basis representation using the `:coordinate`
(default) or `:moving` frame.

For each slot, the corresponding [`BasisElement`](@ref) is constructed
from the registered frame of the given `type`, with the index flipped
to the dual vbundle:

    basis_expansion(T[-a1,-a2])                  # → T[-a1,-a2] dx[a1] ⊗ dx[a2]
    basis_expansion(T[-a1,-a2]; frame=:moving)   # → T[-a1,-a2] θ[a1]  ⊗ θ[a2]
    basis_expansion(T[a1, a2])                   # → T[a1,a2]   ∂[-a1] ⊗ ∂[-a2]
    basis_expansion(T[a1,-a2])                   # → T[a1,-a2]  ∂[-a1] ⊗ dx[a2]

Errors if no frame of the requested type has been registered for any slot's bundle.
"""
function basis_expansion(te::TensorExpression; frame::Symbol=:coordinate)
    manifold_sym = te.tensor.manifold
    haskey(_MANIFOLDS, manifold_sym) ||
        error("basis_expansion: tensor references unregistered manifold :$manifold_sym.")

    basis_elements = BasisElement[]
    for (slot_vb, idx) in zip(te.tensor.slots, te.indices)
        key = (slot_vb, frame)
        haskey(_BASES, key) ||
            error(
                "basis_expansion: no $(frame) frame registered for vbundle :$slot_vb. " *
                "Call @def_manifold or @def_frame_bundle for this bundle first."
            )
        b        = _BASES[key]
        dual_vb  = _VBUNDLES[slot_vb].dual
        dual_idx = idx isa CoordinateIndex ?
            CoordinateIndex(idx.symbol, dual_vb) :
            BasisIndex(idx.symbol, dual_vb)
        push!(basis_elements, BasisElement(b, dual_idx))
    end
    BasisExpansion(te, basis_elements)
end

"""
    basis_expansion(T::Tensor; frame::Symbol=:coordinate) -> BasisExpansion

Expand `T` using canonical slot indices (the first `rank(T)` indices
from the manifold's tangent bundle, each assigned the slot's vbundle).
Delegates to `basis_expansion(::TensorExpression; frame=frame)`.
"""
function basis_expansion(T::Tensor; frame::Symbol=:coordinate)
    haskey(_MANIFOLDS, T.manifold) ||
        error("basis_expansion: tensor references unregistered manifold :$(T.manifold).")
    m          = _MANIFOLDS[T.manifold]
    tb_coord = _VBUNDLES[m.tangent_bundle].coordinate_indices
    n          = T.rank
    n <= length(tb_coord) ||
        error(
            "basis_expansion: tensor rank $n exceeds number of registered coordinate indices " *
            "($(length(tb_coord))). Add more with @add_indices."
        )
    canonical_idxs = [
        CoordinateIndex(tb_coord[i].symbol, T.slots[i])
        for i in 1:n
    ]
    basis_expansion(TensorExpression(T, canonical_idxs); frame=frame)
end


# =========================================
# 13.  show — Basis
# =========================================

function Base.show(io::IO, b::Basis)
    type_label = b.type === :coordinate ? "coordinate" : "moving"
    print(io, "Basis($(b.name), $(b.vbundle), $(type_label))")
end


# =========================================
# 14.  show — FrameBundle
# =========================================

function Base.show(io::IO, fb::FrameBundle)
    print(io, "FrameBundle($(fb.name), vbundle=$(fb.vbundle), dual=$(fb.dual), basis=$(fb.basis.name))")
end

function Base.show(io::IO, ::MIME"text/html", fb::FrameBundle)
    print(io, """
    <div style="border:1px solid #ddd;padding:10px;border-radius:5px;background:#f4faff;">
        <h4 style="margin-top:0;">FrameBundle: <span style="color:#0d6efd;">$(fb.name)</span></h4>
        <table style="width:100%;border-collapse:collapse;">
            <tr><td style="font-weight:bold;width:150px;">VBundle</td><td><code>$(fb.vbundle)</code></td></tr>
            <tr><td style="font-weight:bold;">Dual</td><td><code>$(fb.dual)</code></td></tr>
            <tr><td style="font-weight:bold;">Moving basis</td><td><code>$(fb.basis.name)</code></td></tr>
        </table>
    </div>
    """)
end


# =========================================
# 15.  show — BasisElement
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
# 16.  show — BasisExpansion
# =========================================

"""
    Base.show(io::IO, bx::BasisExpansion)

REPL display: component followed by basis elements joined with ⊗.
No ⊗ between the component and the first basis element.

    T[-a1,-a2] dx[a1] ⊗ dx[a2]
    T[-a1,-a2] θ[a1]  ⊗ θ[a2]
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

    T_{a_{1} a_{2}}\\,\\mathrm{d}x^{a_{1}} \\otimes \\mathrm{d}x^{a_{2}}
"""
function Base.show(io::IO, ::MIME"text/latex", bx::BasisExpansion)
    comp_str  = sprint(show, MIME"text/latex"(), bx.component)
    comp_core = strip(comp_str, ['$'])
    basis_strs = [_format_basis_element_latex(be) for be in bx.basis_elements]
    result = comp_core
    if !isempty(basis_strs)
        result *= "\\," * join(basis_strs, " \\otimes ")
    end
    print(io, "\$", result, "\$")
end

"""
    Base.show(io::IO, ::MIME"text/html", bx::BasisExpansion)

HTML display using `<sup>` / `<sub>` tags.
The component uses `_format_html` from `tensorExpressions.jl`.
"""
function Base.show(io::IO, ::MIME"text/html", bx::BasisExpansion)
    comp_html = _format_html(bx.component)
    if isempty(bx.basis_elements)
        print(io, comp_html)
        return
    end
    basis_html = join(
        [_format_basis_element_html(be) for be in bx.basis_elements],
        " &#x2297; "
    )
    print(io, comp_html, " ", basis_html)
end


# =========================================
# Exports
# =========================================

export Basis, BasisElement, BasisExpansion, FrameBundle
export _BASES, _FRAME_BUNDLES
export basis_for_vbundle, bases_for_vbundle
export @def_frame_bundle, @undef_frame_bundle
export basis_expansion
