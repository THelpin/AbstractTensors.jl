# =========================================
# tensors.jl — AbstractTensors.jl
#
# Design principles (mirrors manifolds.jl / indices.jl):
#   - A Tensor is a plain struct instance. @def_tensor binds the tensor
#     variable in the caller's scope and registers it in _TENSORS.
#   - Slots store vbundle names only — not the defining index symbols.
#     The symbols in @def_tensor T[-a1, a2] M are used only for validation
#     (checking manifold membership) and then discarded.
#   - Symmetries is a Vector{SlotSymmetry} acting on positions 1:n.
#   - Metric is stored as a Symbol key into _METRICS, or nothing.
#   - Dot-access provides derived properties (rank, manifold_data).
#
# xTensor analogs:
#   DefTensor[T[-i1,-i2], M]  →  @def_tensor T[-a1, -a2] M
#   $Tensors                  →  _TENSORS
#   TensorQ[T]                →  is_tensor(T)
#   RankOfTensor[T]           →  T.rank
#   SlotsOfTensor[T]          →  T.slots
#   SymmetryGroupOfTensor[T]  →  T.symmetries
#   PrintAs[T]                →  T.print_as
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
    T.manifold_data  # the Manifold instance (derived — looks up _MANIFOLDS)

Fields
------
- `manifold`      : name of the base manifold, key into `_MANIFOLDS`
- `slots`         : vbundle symbol per slot, encoding variance.
                    `:cotangentM` → covariant, `:tangentM` → contravariant.
                    Built from the index signs in the `@def_tensor` expression.
- `symmetries`    : `Vector{`[`SlotSymmetry`](@ref)`}` — list of permutation
                    groups acting on slot positions `1:rank`. Use convenience
                    constructors `symmetric`, `antisymmetric`, etc.
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
    elseif field === :manifold_data
        m = getfield(t, :manifold)
        haskey(_MANIFOLDS, m) || error("Tensor references unregistered manifold :$m")
        return _MANIFOLDS[m]
    else
        return getfield(t, field)
    end
end

function Base.propertynames(::Tensor, private::Bool=false)
    (:manifold, :slots, :symmetries, :is_traceless, :known_traces, :print_as,
     :metric, :rank, :manifold_data)
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

"""
    rank_of(T::Tensor) -> Int

Number of slots of `T`. Equivalent to `T.rank` and `length(T.slots)`.
"""
rank_of(T::Tensor) = length(T.slots)

"""
    manifold_of(T::Tensor) -> Symbol

Name of the base manifold of `T` (key into `_MANIFOLDS`).
"""
manifold_of(T::Tensor) = T.manifold

"""
    slots_of(T::Tensor) -> Vector{Symbol}

Vbundle symbol per slot, e.g. `[:cotangentM, :tangentM]`.
"""
slots_of(T::Tensor) = T.slots

"""
    symmetries_of(T::Tensor) -> Vector{SlotSymmetry}
"""
symmetries_of(T::Tensor) = T.symmetries

"""
    is_traceless_tensor(T::Tensor) -> Bool

`true` if `T` was declared traceless.
"""
is_traceless_tensor(T::Tensor) = T.is_traceless

"""
    metric_of(T::Tensor) -> Union{Symbol, Nothing}

Return the name of the metric associated with `T`, or `nothing` if no metric
was assigned at definition time.
"""
metric_of(T::Tensor) = T.metric




# =========================================
# 4.  Macro helpers  (expansion-time only)
# =========================================

# Parse T[-a1, a2, -a3] → (:T, [(:a1,true), (:a2,false), (:a3,true)])
# Called at macro expansion time; errors early with clear messages.
function _parse_tensor_head(expr)
    Meta.isexpr(expr, :ref) ||
        error("@def_tensor: first argument must be T[...] syntax, got: $expr")

    tensor_name = expr.args[1]
    tensor_name isa Symbol ||
        error("@def_tensor: tensor name must be a plain symbol, got: $tensor_name")

    slot_specs = Tuple{Symbol, Bool}[]
    for arg in expr.args[2:end]
        if arg isa Symbol
            push!(slot_specs, (arg, false))   # contravariant: up index
        elseif Meta.isexpr(arg, :call) &&
               length(arg.args) == 2   &&
               arg.args[1] == :-       &&
               arg.args[2] isa Symbol
            push!(slot_specs, (arg.args[2], true))   # covariant: down index
        else
            error(
                "@def_tensor: each slot must be a plain symbol (contravariant) " *
                "or -symbol (covariant), got: $arg"
            )
        end
    end

    isempty(slot_specs) &&
        error("@def_tensor: tensor must have at least one slot")

    return tensor_name, slot_specs
end

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
            # Accept bare symbol g or quoted :g — always embed as the name Symbol.
            if v isa Symbol
                metric_expr = QuoteNode(v)        # metric=g  → :g
            elseif v isa QuoteNode && v.value isa Symbol
                metric_expr = v                   # metric=:g → :g
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
    @def_tensor name[slot1, slot2, ...] manifold
    @def_tensor name[slot1, slot2, ...] manifold  symmetries=[S1,...]  traceless=bool  print_as=:sym  metric=g

Define a new abstract tensor on `manifold` and bind it to `name` in the
caller's scope.

Slot syntax
-----------
- `-idx` : covariant (lower) slot — index lives in `cotangentM`
-  `idx` : contravariant (upper) slot — index lives in `tangentM`

The index symbols (`idx`) must already be registered to `manifold` via
`@def_manifold` or `@add_indices`. They are used only for validation;
the tensor stores vbundle names per slot, not the defining symbols.

Keyword arguments (all optional)
---------------------------------
- `symmetries`  : a [`SlotSymmetry`](@ref) or `Vector{SlotSymmetry}` describing
                  the slot permutation symmetry group(s). Defaults to
                  `[no_symmetry(rank)]`. Use `symmetric(2)`, `antisymmetric(2)`,
                  `symmetric_on(4, [1,2])`, `riemann_symmetry()`, etc.
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

# Examples
```julia
@def_manifold M 4 [a1, a2, a3, a4]
@def_metric g[-a1, -a2] M

@def_tensor T[-a1, -a2] M                                   # metric auto-resolved to :g
@def_tensor F[-a1, -a2] M symmetries=[antisymmetric(2)]
@def_tensor R[-a1,-a2,-a3,-a4] M symmetries=[riemann_symmetry()]
@def_tensor W[-a1,-a2,-a3,-a4] M symmetries=[riemann_symmetry()] traceless=true print_as=:Weyl
@def_tensor mixed_T[a1, -a2] M   # mixed (1,1) tensor
```
"""
macro def_tensor(tensor_expr, manifold_expr, kwargs...)
    # ── Expansion-time parsing ─────────────────────────────────────────
    manifold_expr isa Symbol ||
        error("@def_tensor: second argument must be a manifold symbol, got: $manifold_expr")

    tensor_name, slot_specs = _parse_tensor_head(tensor_expr)
    symmetries_expr, traceless, print_as_sym, metric_expr =
        _parse_tensor_kwargs(kwargs, tensor_name)

    n             = length(slot_specs)
    index_syms    = [s for (s, _) in slot_specs]
    is_cov_flags  = [c for (_, c) in slot_specs]

    manifold_sym  = QuoteNode(manifold_expr)
    tensor_sym    = QuoteNode(tensor_name)
    print_as_node = QuoteNode(print_as_sym)

    idx_syms_expr  = :([$(map(QuoteNode, index_syms)...)])
    cov_flags_expr = :([$(is_cov_flags...)])

    # Symmetries: default to [no_symmetry(n)]; normalize to Vector{SlotSymmetry}
    # at runtime.
    sym_expr = isnothing(symmetries_expr) ? :([no_symmetry($n)]) : esc(symmetries_expr)

    # Metric: nothing means "auto-resolve at runtime from _METRICS".
    # metric_expr is always a QuoteNode(:sym) when provided (set in _parse_tensor_kwargs).
    metric_provided = !isnothing(metric_expr)
    metric_val_expr = metric_provided ? metric_expr : nothing

    quote
        # ── Guard: warn if redefining ──────────────────────────────────
        if haskey(_TENSORS, $(tensor_sym))
            @warn "Tensor $($(tensor_sym)) is already defined. Redefining."
        end

        # ── Validate manifold ──────────────────────────────────────────
        haskey(_MANIFOLDS, $(manifold_sym)) ||
            error(
                "@def_tensor: manifold $($(manifold_sym)) is not registered. " *
                "Call @def_manifold $($(manifold_sym)) first."
            )
        local _M = _MANIFOLDS[$(manifold_sym)]

        # ── Validate indices ───────────────────────────────────────────
        validate_indices($(idx_syms_expr), _M.tangent_bundle)

        # ── Build slots vector ─────────────────────────────────────────
        local _is_cov = $(cov_flags_expr)
        local _slots  = Symbol[
            _is_cov[i] ? _M.cotangent_bundle : _M.tangent_bundle
            for i in 1:$n
        ]

        # ── Evaluate and normalize symmetries ──────────────────────────
        local _raw_sym = $sym_expr
        local _syms::Vector{SlotSymmetry} = if _raw_sym isa SlotSymmetry
            [_raw_sym]
        elseif _raw_sym isa AbstractVector
            Vector{SlotSymmetry}(_raw_sym)
        else
            error(
                "@def_tensor: symmetries must be a SlotSymmetry or " *
                "Vector{SlotSymmetry}, got $(typeof(_raw_sym))."
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
            # Explicit metric= supplied: _given_sym is always a Symbol (enforced
            # in _parse_tensor_kwargs by wrapping bare symbols in QuoteNode).
            local _given_sym::Symbol = $(metric_val_expr)
            haskey(_METRICS, _given_sym) ||
                error(
                    "@def_tensor: $($(tensor_sym)) metric= :$(_given_sym) is not " *
                    "registered. Call @def_metric :$(_given_sym) first."
                )
            _METRICS[_given_sym] == $(manifold_sym) ||
                error(
                    "@def_tensor: metric :$(_given_sym) belongs to manifold " *
                    "$(_METRICS[_given_sym]), but tensor " *
                    "$($(tensor_sym)) is on $($(manifold_sym))."
                )
            _metric_sym = _given_sym
        else
            # Auto-resolve from _METRICS.
            local _known = metrics_of_manifold($(manifold_sym))
            if isempty(_known)
                @warn "No metric is defined on manifold $($(manifold_sym)). " *
                      "Tensor $($(tensor_sym)) has no metric assigned; " *
                      "indices cannot be raised or lowered."
                _metric_sym = nothing
            elseif length(_known) == 1
                _metric_sym = _known[1]
            else
                @warn "Multiple metrics $(tuple(_known...)) are defined on " *
                      "manifold $($(manifold_sym)). Assigning first " *
                      "(:$(_known[1])) to tensor $($(tensor_sym)). " *
                      "Use metric=<name> to be explicit."
                _metric_sym = _known[1]
            end
        end

        # ── Construct and register ─────────────────────────────────────
        local _T = Tensor(
            $(manifold_sym),
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
            "Defined tensor $($(tensor_sym)) on manifold $($(manifold_sym)) " *
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

    @def_tensor T[-a1, -a2] M
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

function Base.show(io::IO, T::Tensor)
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
            <tr><td style="font-weight:bold;width:120px;">Manifold</td>
                <td><code>$(T.manifold)</code></td></tr>
            <tr><td style="font-weight:bold;">Rank</td>
                <td>$(T.rank)</td></tr>
            <tr><td style="font-weight:bold;">Slots</td>
                <td>$(join(slot_html, " &nbsp; "))</td></tr>
            <tr><td style="font-weight:bold;">Symmetries</td>
                <td>$(sym_strs)</td></tr>
            <tr><td style="font-weight:bold;">Traceless</td>
                <td>$(T.is_traceless)</td></tr>
            <tr><td style="font-weight:bold;">Metric</td>
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
export is_tensor, rank_of, manifold_of, slots_of, symmetries_of
export is_traceless_tensor, metric_of
export list_tensors, tensor_info
export @def_tensor, @undef_tensor
