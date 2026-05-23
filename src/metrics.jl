# =========================================
# metrics.jl — SymbolicTensors.jl
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
#   DefMetric[g[-i,-j], M]  →  @def_metric g M
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
    @def_metric name M

Define a new metric tensor on manifold `M`, bind it to `name` in the
caller's scope, and register it in both [`_TENSORS`](@ref) and
[`_METRICS`](@ref).

`@def_metric` is a specialised shortcut (not a call to [`@def_tensor`](@ref))
that always builds a rank-2 fully covariant symmetric metric:

- Slots: `[cotangentM, cotangentM]` from `M.cotangent_bundle` (both covariant).
- `symmetries=[symmetric(2)]` — fixed, no user override.
- `print_as` set to `string(name)`; `metric` set to `name` (self-referential).
- No keyword arguments.

At **expression** time you still write `g[-a1, -a2]` using coordinate indices;
only **definition** uses `@def_metric g M`.

# Examples
~~~julia
@def_manifold M 4 [a1, a2, a3, a4] [A1, A2, A3, A4]

@def_metric g M    # Riemannian metric on M
@def_metric η M    # second metric on the same manifold
~~~
"""
macro def_metric(name, manifold_expr)
    name isa Symbol ||
        error("@def_metric: first argument must be a symbol, got: $name")
    manifold_expr isa Symbol ||
        error("@def_metric: second argument must be a manifold symbol, got: $manifold_expr")

    manifold_sym = QuoteNode(manifold_expr)
    name_sym     = QuoteNode(name)
    print_as_str = string(name)

    quote
        # ── Validate manifold ─────────────────────────────────────────
        haskey(_MANIFOLDS, $(manifold_sym)) ||
            error("@def_metric: manifold $($(manifold_sym)) is not registered")

        local _M = _MANIFOLDS[$(manifold_sym)]
        local _cotangent = _M.cotangent_bundle

        # ── Guard: warn if redefining ─────────────────────────────────
        if haskey(_TENSORS, $(name_sym))
            @warn "Metric $($(name_sym)) is already defined. Redefining."
        end
        if haskey(_METRICS, $(name_sym))
            delete!(_METRICS, $(name_sym))
        end

        # ── Build metric tensor ────────────────────────────────────────
        local _slots = [_cotangent, _cotangent]

        # Metrics are symmetric (0,2) tensors
        local _syms = [symmetric(2)]

        local _T = Tensor(
            $(manifold_sym),    # manifold
            _slots,             # slots = [cotangentM, cotangentM]
            _syms,              # symmetries = [symmetric(2)]
            false,              # is_traceless
            Any[],              # known_traces
            $(QuoteNode(print_as_str)),  # print_as
            $(name_sym)                 # metric (self-referential)
        )

        # ── Register ─────────────────────────────────────────────────
        _TENSORS[$(name_sym)] = _T
        _METRICS[$(name_sym)] = $(manifold_sym)
        $(esc(name)) = _T

        println("Defined metric $($(name_sym)) on manifold $($(manifold_sym))")
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

    @def_metric g M
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
        haskey(_VBUNDLES, vb) ? (_VBUNDLES[vb].isref ? "↑" : "↓") : "?"
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
