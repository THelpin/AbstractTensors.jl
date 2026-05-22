# =========================================
# tensors.jl — SymbolicTensors.jl
#
# Design principles:
#   - A Tensor is a plain struct instance. @def_tensor binds the tensor
#     variable in the caller's scope and registers it in _TENSORS.
#   - Slots store vbundle names only.
#   - Symmetries is a Vector{SlotSymmetry} acting on positions 1:n.
#   - Metric is stored as a Symbol key into _METRICS, or nothing.
#   - Dot-access provides derived properties (rank).
# =========================================


# =========================================
# 1.  Tensor struct
# =========================================

"""
    Tensor

A registered abstract tensor. Instances are created by [`@def_tensor`](@ref)
and bound to a variable in the caller's scope.

Provides dot access to all metadata:

    T.manifold       # :M  (Symbol key into _MANIFOLDS)
    T.slots          # [:cotangentM, :cotangentM]  — vbundle per slot
    T.symmetries     # [SlotSymmetry(n=2, order=2)]
    T.is_traceless   # false
    T.known_traces   # Any[]  (populated later)
    T.print_as       # :T
    T.metric         # :g or nothing
    T.rank           # 2      (derived — length of slots)

### Fields

- `manifold`      : name of the base manifold, key into `_MANIFOLDS`
- `slots`         : vbundle symbol per slot, encoding variance.
                    `:cotangentM` → covariant, `:tangentM` → contravariant.
- `symmetries`    : `Vector{`[`SlotSymmetry`](@ref)`}` — list of permutation
                    groups acting on slot positions `1:rank`.
- `is_traceless`  : if `true`, any self-contraction of this tensor gives zero
                    (e.g. the Weyl tensor).
- `known_traces`  : user-declared trace values, e.g. `g[a,-a] = dim`.
                    Format TBD — stored as `Any[]` until contraction is
                    implemented.
- `print_as`      : display name. Controls how the tensor appears in `show`
                    and (later) LaTeX output.
- `metric`        : name of the metric tensor associated with this tensor,
                    a key into `_METRICS`, or `nothing` if no metric
                    has been assigned. Required for raising/lowering indices.
"""
struct Tensor
    manifold::Symbol
    slots::Vector{Symbol}
    symmetries::Vector{SlotSymmetry}
    is_traceless::Bool
    known_traces::Vector{Any}
    print_as::Symbol
    metric::Union{Symbol, Nothing}
end

# ── Derived property access ────────────────────────────────────────────────

function Base.getproperty(t::Tensor, field::Symbol)
    if field === :rank
        return length(getfield(t, :slots))
    else
        return getfield(t, field)
    end
end

function Base.propertynames(::Tensor, private::Bool=false)
    (:manifold, :slots, :symmetries, :is_traceless, :known_traces, :print_as,
     :metric, :rank)
end


# =========================================
# 2.  Module-level registries
# =========================================

"""
    _TENSORS :: Dict{Symbol, Tensor}

Maps each registered tensor name to its [`Tensor`](@ref) instance.

    _TENSORS[:T]  →  Tensor(...)

Populated by `@def_tensor` (and `@def_metric`), cleared entry-by-entry by
`@undef_tensor` (and `@undef_metric`).
Do not mutate directly — use the macro API.
"""
const _TENSORS = Dict{Symbol, Tensor}()



# =========================================
# 3.  Accessors, predicates, and helpers
# =========================================

"""
    is_tensor(x) -> Bool

Return `true` if `x` is a [`Tensor`](@ref) instance.
"""
is_tensor(x) = x isa Tensor


# =========================================
# 4.  Macro helpers  (expansion-time only)
# =========================================

# Parse keyword arguments symmetries=..., traceless=..., print_as=..., metric=...
# Returns (symmetries_expr_or_nothing, traceless::Bool, print_as::Symbol, metric_expr_or_nothing)
function _parse_tensor_kwargs(kwargs, default_print_as::Symbol)
    symmetries_expr = nothing
    traceless       = false
    print_as_sym    = default_print_as
    metric_expr     = nothing

    for kw in kwargs
        Meta.isexpr(kw, :(=), 2) ||
            error("@def_tensor: expected keyword=value argument, got: $kw")
        k, v = kw.args

        if k === :symmetries
            symmetries_expr = v
        elseif k === :traceless
            v isa Bool ||
                error("@def_tensor: traceless must be a literal true or false, got: $v")
            traceless = v
        elseif k === :print_as
            if v isa QuoteNode && v.value isa Symbol
                print_as_sym = v.value
            elseif v isa Symbol
                print_as_sym = v
            else
                error("@def_tensor: print_as must be a quoted symbol, e.g. print_as=:Riemann")
            end
        elseif k === :metric
            if v isa Symbol
                metric_expr = QuoteNode(v)
            elseif v isa QuoteNode && v.value isa Symbol
                metric_expr = v
            else
                error("@def_tensor: metric= must be a symbol name, e.g. metric=g")
            end
        else
            error(
                "@def_tensor: unknown keyword :$k. " *
                "Supported keywords: symmetries, traceless, print_as, metric."
            )
        end
    end

    return symmetries_expr, traceless, print_as_sym, metric_expr
end


# =========================================
# 5.  @def_tensor macro
# =========================================

"""
    @def_tensor name [vbundle1, vbundle2, ...] [symmetries=...] [traceless=...] [print_as=:...] [metric=...]

Define a new abstract tensor and bind it to `name` in the caller's scope.

### Slot syntax

Specify slots directly as VBundle symbols. The manifold is automatically
deduced from the first VBundle's `manifold` field.

- `:tangentM` → contravariant (upper) slot
- `:cotangentM` → covariant (lower) slot
- Any user-defined VBundle from `@def_vbundle`

All VBundle symbols must belong to the same manifold.

### Keyword arguments (all optional)

- `symmetries`  : a [`SlotSymmetry`](@ref) or `Vector{SlotSymmetry}` describing
                  the slot permutation symmetry group(s). Defaults to
                  `[no_symmetry(rank)]`.
- `traceless`   : `true` if any self-contraction of this tensor is zero
                  (e.g. Weyl tensor). Defaults to `false`.
- `print_as`    : display name. Defaults to `name`. Example: `print_as=:R`.
- `metric`      : name of the metric tensor to associate with this tensor.
                  Omitting this keyword triggers automatic resolution:
                  - one metric on manifold → assigned silently
                  - multiple metrics → `@warn`, first defined is assigned
                  - no metric → `@warn`, `nothing` assigned (no raising/lowering)

Binds `name` to a [`Tensor`](@ref) instance in the caller's scope and
registers it in [`_TENSORS`](@ref).

### Examples

~~~julia
@def_manifold M 4 [a1, a2, a3, a4]
@def_metric g M

@def_tensor T [cotangentM, cotangentM]                 # (0,2) tensor, metric auto-resolved
@def_tensor F [cotangentM, cotangentM] symmetries=[antisymmetric(2)]
@def_tensor R [cotangentM, cotangentM, cotangentM, tangentM] symmetries=[riemann_symmetry()]
@def_tensor W [cotangentM, cotangentM, cotangentM, tangentM] symmetries=[riemann_symmetry()] traceless=true print_as=:Weyl
@def_tensor mixed_T [tangentM, cotangentM]            # (1,1) mixed tensor
~~~
"""
macro def_tensor(tensor_name, vbundle_list, kwargs...)
    tensor_name isa Symbol ||
        error("@def_tensor: first argument must be a tensor name symbol, got: $tensor_name")

    Meta.isexpr(vbundle_list, :vect) ||
        error("@def_tensor: second argument must be a vector, got: $vbundle_list")

    # Build expressions to extract VBundle names at runtime
    name_exprs = []
    for arg in vbundle_list.args
        if arg isa Symbol
            push!(name_exprs, QuoteNode(arg))
        elseif arg isa QuoteNode && arg.value isa Symbol
            push!(name_exprs, arg)
        else
            push!(name_exprs, :( $(esc(arg)).name ))
        end
    end

    isempty(name_exprs) &&
        error("@def_tensor: tensor must have at least one slot")

    n = length(name_exprs)

    symmetries_expr, traceless, print_as_sym, metric_expr =
        _parse_tensor_kwargs(kwargs, tensor_name)

    # Handle default symmetries: if not provided, use [no_symmetry(n)]
    if isnothing(symmetries_expr)
        symmetries_expr = :([no_symmetry($n)])
    else
        symmetries_expr = esc(symmetries_expr)
    end

    tensor_sym = QuoteNode(tensor_name)
    print_as_node = QuoteNode(print_as_sym)
    slots_expr = :([$(name_exprs...)])

    metric_provided = !isnothing(metric_expr)
    metric_val_expr = metric_provided ? metric_expr : nothing

    quote
        # ── Guard: warn if redefining ──────────────────────────────────
        if haskey(_TENSORS, $(tensor_sym))
            @warn "Tensor $($(tensor_sym)) is already defined. Redefining."
        end

        # ── Extract VBundle names at runtime ────────────────────────────
        local _slots = $(slots_expr)

        # ── Validate VBundle symbols ───────────────────────────────────
        for vb in _slots
            haskey(_VBUNDLES, vb) ||
                error(
                    "@def_tensor: VBundle $vb is not registered. " *
                    "Call @def_manifold or @def_vbundle first."
                )
        end

        # ── Deduce manifold from first VBundle ─────────────────────────
        local _manifold_sym = _VBUNDLES[_slots[1]].manifold

        # ── Validate all VBundle symbols belong to same manifold ───────
        for vb in _slots
            _VBUNDLES[vb].manifold == _manifold_sym || error(
                "@def_tensor: VBundle $vb belongs to manifold $(_VBUNDLES[vb].manifold), " *
                "but first VBundle belongs to $(_manifold_sym)"
            )
        end

        # ── Evaluate and normalize symmetries ──────────────────────────
        local _raw_sym = $symmetries_expr
        local _syms::Vector{SlotSymmetry}
        if _raw_sym isa AbstractVector
            _syms = Vector{SlotSymmetry}(_raw_sym)
        else
            error(
                "@def_tensor: symmetries must be a Vector{SlotSymmetry}. " *
                "Use symmetries=[...] syntax, got $(typeof(_raw_sym))."
            )
        end

        for (k, _s) in enumerate(_syms)
            _s isa SlotSymmetry ||
                error(
                    "@def_tensor: symmetries[$k] is not a SlotSymmetry, " *
                    "got $(typeof(_s))."
                )
            _s.n == $n ||
                error(
                    "@def_tensor: symmetries[$k] has degree $(_s.n) but tensor " *
                    "$($(tensor_sym)) has $($n) slot(s)."
                )
        end

        # ── Resolve metric ─────────────────────────────────────────────
        local _metric_sym::Union{Symbol, Nothing}
        if $(metric_provided)
            local _given_sym::Symbol = $(metric_val_expr)
            haskey(_METRICS, _given_sym) ||
                error(
                    "@def_tensor: $($(tensor_sym)) metric= :$(_given_sym) is not " *
                    "registered. Call @def_metric :$(_given_sym) first."
                )
            _METRICS[_given_sym] == _manifold_sym ||
                error(
                    "@def_tensor: metric :$(_given_sym) belongs to manifold " *
                    "$(_METRICS[_given_sym]), but tensor " *
                    "$($(tensor_sym)) is on $_manifold_sym."
                )
            _metric_sym = _given_sym
        else
            local _known = metrics_of_manifold(_manifold_sym)
            if isempty(_known)
                @warn "No metric is defined on manifold $_manifold_sym. " *
                      "Tensor $($(tensor_sym)) has no metric assigned; " *
                      "indices cannot be raised or lowered."
                _metric_sym = nothing
            elseif length(_known) == 1
                _metric_sym = _known[1]
            else
                @warn "Multiple metrics $(tuple(_known...)) are defined on " *
                      "manifold $_manifold_sym. Assigning first " *
                      "(:$(_known[1])) to tensor $($(tensor_sym)). " *
                      "Use metric=<name> to be explicit."
                _metric_sym = _known[1]
            end
        end

        # ── Construct and register ─────────────────────────────────────
        local _T = Tensor(
            _manifold_sym,
            _slots,
            _syms,
            $(traceless),
            Any[],
            $(print_as_node),
            _metric_sym
        )

        _TENSORS[$(tensor_sym)] = _T
        $(esc(tensor_name)) = _T

        println(
            "Defined tensor $($(tensor_sym)) on manifold $_manifold_sym " *
            "with $($n) slot(s), metric=$(_metric_sym)"
        )
        nothing
    end
end

# =========================================
# 6.  @undef_tensor macro
# =========================================

"""
    @undef_tensor name

Remove a non-metric tensor from [`_TENSORS`](@ref).

If `name` is registered as a metric in [`_METRICS`](@ref), this macro errors
and does not modify any registry. Use [`@undef_metric`](@ref) instead to
remove a metric (that macro clears both [`_METRICS`](@ref) and
[`_TENSORS`](@ref)).

Note that any tensor that referenced a removed metric will still hold the
stale `metric` symbol in its struct; re-define those tensors if needed.

The variable `name` in the caller's scope still holds the old [`Tensor`](@ref)
struct after this call (Julia variables cannot be un-bound by a macro), but
the tensor is no longer reachable via the registries or `list_tensors`.
"""
macro undef_tensor(name)
    name isa Symbol ||
        error("@undef_tensor: argument must be a symbol, got: $name")
    name_sym = QuoteNode(name)
    quote
        haskey(_TENSORS, $(name_sym)) ||
            error(
                "@undef_tensor: tensor $($(name_sym)) is not registered. " *
                "Call @def_tensor $($(name_sym)) first, or check list_tensors()."
            )
        haskey(_METRICS, $(name_sym)) &&
            error(
                "@undef_tensor: $($(name_sym)) is a metric, not a plain tensor. " *
                "Use `@undef_metric $($(name_sym))` instead."
            )
        delete!(_TENSORS, $(name_sym))
        println("Undefined tensor: $($(name_sym))")
        nothing
    end
end


# =========================================
# 7.  Introspection utilities
# =========================================

"""
    list_tensors() -> Vector{Symbol}

Return the names of all currently registered tensors.

    @def_tensor T [cotangentM, cotangentM]
    list_tensors()   # [:T]
"""
list_tensors() = collect(keys(_TENSORS))

"""
    tensor_info(T::Tensor)

Print a human-readable summary of tensor `T`.
"""
function tensor_info(T::Tensor)
    slot_strs = map(T.slots) do vb
        if haskey(_VBUNDLES, vb)
            _VBUNDLES[vb].isdual ? "-$(vb)" : "+$(vb)"
        else
            "?$(vb)"
        end
    end

    metric_str = T.metric === nothing ? "none" : string(T.metric)

    println("Tensor:      $(T.print_as)")
    println("  Manifold:  $(T.manifold)")
    println("  Rank:      $(T.rank)")
    println("  Slots:     [$(join(slot_strs, ", "))]")
    println("  Symmetries: $(T.symmetries)")
    println("  Traceless: $(T.is_traceless)")
    println("  Metric:    $(metric_str)")
end


# =========================================
# 8.  show methods
# =========================================

function Base.show(io::IO, ::MIME"text/plain", T::Tensor)
    slot_chars = map(T.slots) do vb
        haskey(_VBUNDLES, vb) ? (_VBUNDLES[vb].isdual ? "↓" : "↑") : "?"
    end
    metric_str = T.metric === nothing ? "none" : string(T.metric)
    print(io,
        "Tensor $(T.print_as)[$(join(slot_chars, ""))] " *
        "on $(T.manifold), $(T.symmetries), metric=$(metric_str)"
    )
end

function Base.show(io::IO, ::MIME"text/html", T::Tensor)
    slot_html = map(T.slots) do vb
        if haskey(_VBUNDLES, vb)
            _VBUNDLES[vb].isdual ? "<b>↓</b><code>$vb</code>" : "<b>↑</b><code>$vb</code>"
        else
            "<code>?$vb</code>"
        end
    end

    sym_strs = join(map(s -> "<code>$s</code>", T.symmetries), ", ")
    metric_str = T.metric === nothing ? "<i>none</i>" : "<code>$(T.metric)</code>"

    print(io, """
    <div style="border:1px solid #ddd;padding:10px;border-radius:5px;background:#f9f9f9;">
        <h4 style="margin-top:0;">Tensor: <span style="color:#0a7c40;">$(T.print_as)</span></h4>
        <table style="width:100%;border-collapse:collapse;">
            <tr><td style="font-weight:bold;width:120px;text-align:left;">Manifold</td>
                <td><code>$(T.manifold)</code></td></tr>
            <tr><td style="font-weight:bold;text-align:left;">Rank</td>
                <td>$(T.rank)</td></tr>
            <tr><td style="font-weight:bold;text-align:left;">Slots</td>
                <td>$(join(slot_html, " &nbsp; "))</td></tr>
            <tr><td style="font-weight:bold;text-align:left;">Symmetries</td>
                <td>$(sym_strs)</td></tr>
            <tr><td style="font-weight:bold;text-align:left;">Traceless</td>
                <td>$(T.is_traceless)</td></tr>
            <tr><td style="font-weight:bold;text-align:left;">Metric</td>
                <td>$(metric_str)</td></tr>
        </table>
    </div>
    """)
end


# =========================================
# Exports
# =========================================

export Tensor
export _TENSORS, _METRICS
export is_tensor
export list_tensors, tensor_info
export @def_tensor, @undef_tensor