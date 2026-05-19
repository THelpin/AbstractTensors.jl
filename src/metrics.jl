# =========================================
# metrics.jl — AbstractTensors.jl
#
# A metric is a Tensor that is also registered in _METRICS.
# @def_metric enforces:
#   - rank 2, both slots covariant
#   - symmetries=[symmetric(2)] (imposed automatically)
#   - registration in both _TENSORS and _METRICS
#
# A manifold can have zero, one, or multiple metrics. Individual tensors
# store a reference to whichever metric they were assigned (or nothing).
#
# xTensor analogs:
#   DefMetric[g[-i,-j], M]  →  @def_metric g[-a1,-a2] M
#   $Metrics                →  _METRICS
#   MetricQ[g]              →  is_metric(g)
#   MetricsOfManifold[M]    →  metrics_on_manifold(:M)
# =========================================

# Depends on: indices.jl, manifolds.jl, permutations.jl, tensors.jl

# =========================================
# 0.  Module-level registries
# =========================================

"""
    _METRICS :: Dict{Symbol, Symbol}

Maps each registered metric name to the name of its base manifold.

    _METRICS[:g]  →  :M

Populated by [`@def_metric`](@ref), cleared by [`@undef_metric`](@ref)
(or [`@undef_tensor`](@ref) when the tensor being removed is a metric).
Do not mutate directly — use the macro API.
"""
const _METRICS = Dict{Symbol, Symbol}()



# =========================================
# 1.  @def_metric macro
# =========================================

"""
    @def_metric name[-a1, -a2] M
    @def_metric name[-a1, -a2] M print_as=:sym

Define a new metric tensor on manifold `M`, bind it to `name` in the
caller's scope, and register it in both [`_TENSORS`](@ref) and
[`_METRICS`](@ref).

`@def_metric` is a specialised wrapper around `@def_tensor` that enforces:
- Rank exactly 2.
- Both slots covariant (metric indices are always `g_{ab}`, never mixed).
- `symmetries=[symmetric(2)]` — always fully symmetric, no user override.
- `metric` set to the metric's own name (self-referential: the metric raises
  and lowers its own indices).

Keyword arguments
-----------------
- `print_as` : display symbol. Defaults to `name`.

All other `@def_tensor` keywords (`symmetries`, `traceless`, `metric`) are
intentionally not accepted — they are either fixed by definition or
meaningless for a metric.

# Examples
```julia
@def_manifold M 4 [a1, a2, a3, a4]

@def_metric g[-a1, -a2] M             # Riemannian metric
@def_metric η[-a1, -a2] M print_as=:η # Minkowski / alternative metric
```
"""
macro def_metric(tensor_expr, manifold_expr, kwargs...)
    # ── Expansion-time parsing ─────────────────────────────────────────
    manifold_expr isa Symbol ||
        error("@def_metric: second argument must be a manifold symbol, got: $manifold_expr")

    # Reuse the tensor head parser — gives us name and slot specs.
    tensor_name, slot_specs = _parse_tensor_head(tensor_expr)

    # Only print_as is accepted.
    print_as_sym = tensor_name
    for kw in kwargs
        Meta.isexpr(kw, :(=), 2) ||
            error("@def_metric: expected keyword=value, got: $kw")
        k, v = kw.args
        if k === :print_as
            if v isa QuoteNode && v.value isa Symbol
                print_as_sym = v.value
            elseif v isa Symbol
                print_as_sym = v
            else
                error("@def_metric: print_as must be a quoted symbol, e.g. print_as=:η")
            end
        else
            error(
                "@def_metric: unsupported keyword :$k. " *
                "Only print_as is accepted. Symmetry is always symmetric(2); " *
                "metric is self-referential."
            )
        end
    end

    n = length(slot_specs)
    index_syms   = [s for (s, _) in slot_specs]
    is_cov_flags = [c for (_, c) in slot_specs]

    manifold_sym  = QuoteNode(manifold_expr)
    tensor_sym    = QuoteNode(tensor_name)
    print_as_node = QuoteNode(print_as_sym)

    idx_syms_expr  = :([$(map(QuoteNode, index_syms)...)])
    cov_flags_expr = :([$(is_cov_flags...)])

    quote
        # ── Expansion-time constraints (enforced at runtime) ───────────
        $n == 2 ||
            error(
                "@def_metric: a metric must have exactly 2 slots, got $($n). " *
                "Metric indices are always g_{ab}."
            )
        all($(cov_flags_expr)) ||
            error(
                "@def_metric: both slots of a metric must be covariant (use -idx). " *
                "Got slot pattern: $($(cov_flags_expr)). " *
                "Metrics are always g_{ab}, not g^a{}_b or g^{ab}."
            )

        # ── Guard: warn if redefining ──────────────────────────────────
        if haskey(_METRICS, $(tensor_sym))
            @warn "Metric $($(tensor_sym)) is already defined. Redefining."
        elseif haskey(_TENSORS, $(tensor_sym))
            @warn "Tensor $($(tensor_sym)) is already defined as a non-metric tensor. Redefining as metric."
        end

        # ── Validate manifold ──────────────────────────────────────────
        haskey(_MANIFOLDS, $(manifold_sym)) ||
            error(
                "@def_metric: manifold $($(manifold_sym)) is not registered. " *
                "Call @def_manifold $($(manifold_sym)) first."
            )
        local _M = _MANIFOLDS[$(manifold_sym)]

        # ── Validate indices ───────────────────────────────────────────
        validate_indices($(idx_syms_expr), _M.tangent_bundle)

        # ── Build slots (both cotangent by construction) ───────────────
        local _slots = [_M.cotangent_bundle, _M.cotangent_bundle]

        # ── Construct Tensor ───────────────────────────────────────────
        local _T = Tensor(
            $(manifold_sym),
            _slots,
            [symmetric(2)],
            false,
            Any[],
            $(print_as_node),
            $(tensor_sym)      # self-referential: metric is its own metric
        )

        # ── Register ───────────────────────────────────────────────────
        _TENSORS[$(tensor_sym)]  = _T
        _METRICS[$(tensor_sym)]  = $(manifold_sym)

        $(esc(tensor_name)) = _T

        println(
            "Defined metric $($(tensor_sym)) on manifold $($(manifold_sym))"
        )
        nothing
    end
end


# =========================================
# 2.  @undef_metric macro
# =========================================

"""
    @undef_metric name

Remove a metric from both [`_METRICS`](@ref) and [`_TENSORS`](@ref).

Any other tensor that referenced this metric by name will still hold the stale
symbol in its `metric` field. Redefine those tensors if needed.
"""
macro undef_metric(name)
    name isa Symbol ||
        error("@undef_metric: argument must be a symbol, got: $name")
    name_sym = QuoteNode(name)
    quote
        haskey(_METRICS, $(name_sym)) ||
            error(
                "@undef_metric: metric $($(name_sym)) is not registered. " *
                "Call @def_metric $($(name_sym)) first, or check list_metrics()."
            )
        delete!(_METRICS, $(name_sym))
        haskey(_TENSORS, $(name_sym)) && delete!(_TENSORS, $(name_sym))
        println("Undefined metric: $($(name_sym))")
        nothing
    end
end


# =========================================
# 3.  Predicates and accessors
# =========================================

"""
    metrics_of_manifold(m::Symbol) -> Vector{Symbol}

Return the names of all metrics registered on manifold `m`, in insertion order.
Returns an empty vector if no metrics have been defined on `m`.

Used internally by `@def_tensor` to resolve which metric to assign when the
`metric=` keyword is omitted.
"""
function metrics_of_manifold(m::Symbol)
    # Preserve a deterministic order by sorting names.
    # (Dict iteration order is not guaranteed in Julia.)
    sort([k for (k, v) in _METRICS if v == m])
end

"""
    is_metric(x) -> Bool

Return `true` if `x` is a [`Tensor`](@ref) instance that is also registered
as a metric in [`_METRICS`](@ref).
"""
function is_metric(x)
    x isa Tensor || return false
    # Find the tensor's name by looking it up in _TENSORS.
    for (sym, t) in _TENSORS
        t === x && return haskey(_METRICS, sym)
    end
    return false
end

"""
    list_metrics() -> Vector{Symbol}

Return the names of all currently registered metrics (across all manifolds).

    @def_metric g[-a1, -a2] M
    list_metrics()   # [:g]
"""
list_metrics() = collect(keys(_METRICS))

"""
    metric_info(g::Tensor)

Print a human-readable summary of metric `g`.
"""
function metric_info(g::Tensor)
    is_metric(g) ||
        @warn "metric_info: tensor $(g.print_as) is not registered as a metric."
    println("Metric:    $(g.print_as)")
    println("  Manifold: $(g.manifold)")
    println("  Slots:    [$(join(g.slots, ", "))]")
    println("  Symmetry: $(g.symmetries)")
end


# =========================================
# 4.  show methods
# =========================================

# Metrics use the same Tensor show methods defined in tensors.jl.
# The specialised `show` below is for the REPL one-line representation when
# the user has a reference to a metric and types its name.
# We detect "is a metric" by checking _METRICS at display time.

function _metric_show_line(io::IO, g::Tensor)
    slot_chars = map(g.slots) do vb
        haskey(_VBUNDLES, vb) ? (_VBUNDLES[vb].isdual ? "↓" : "↑") : "?"
    end
    print(io,
        "Metric $(g.print_as)[$(join(slot_chars, ""))] on $(g.manifold)"
    )
end

# We cannot override Base.show(io, ::Tensor) only for metrics without a
# separate type. Instead we export metric_info and let the standard Tensor
# show handle display. If you want a custom REPL line, use metric_info(g).


# =========================================
# 5.  HTML show (Jupyter)
# =========================================

"""
    show_metrics()

Print an HTML summary of all registered metrics.
Intended for Jupyter / IJulia notebooks.
"""
function show_metrics()
    isempty(_METRICS) && (println("No metrics registered."); return)
    for sym in keys(_METRICS)
        haskey(_TENSORS, sym) && display(_TENSORS[sym])
    end
end


# =========================================
# Exports
# =========================================

export _METRICS
export @def_metric, @undef_metric
export is_metric, list_metrics, metric_info, show_metrics, metrics_of_manifold
