# =========================================
# metrics.jl — SymbolicTensors.jl
#
# A metric is a Tensor that is also registered in _METRICS.
# @def_metric enforces:
#   - rank 2, both slots covariant
#   - symmetries=[symmetric(2)] (imposed automatically)
#   - registration in both _TENSORS and _METRICS
#
# A vbundle of reference can have zero, one, or multiple metrics. Individual
# tensors store a reference to whichever metric they were assigned (or nothing).
#
# xTensor analogs:
#   DefMetric[g[-i,-j], tangentM]  →  @def_metric g tangentM
#   $Metrics                       →  _METRICS
#   MetricQ[g]                     →  is_metric(g)
#   MetricsOfManifold[M]             →  metrics_of_manifold(M)
# =========================================

# Depends on: indices.jl, manifolds.jl, permutations.jl, tensors.jl

# =========================================
# 0.  Module-level registries
# =========================================

"""
    _METRICS :: Dict{Symbol, Symbol}

Maps each registered metric name to its vbundle of reference.

    _METRICS[:g]  →  :tangentM

Populated by [`@def_metric`](@ref), cleared by [`@undef_metric`](@ref)
(or [`@undef_tensor`](@ref) when the tensor being removed is a metric).
Do not mutate directly — use the macro API.
"""
const _METRICS = Dict{Symbol, Symbol}()



# =========================================
# 1.  @def_metric macro
# =========================================

"""
    @def_metric name vbundle

Define a new metric tensor for vbundle of reference `vbundle`, bind it to `name`
in the caller's scope, and register it in both [`_TENSORS`](@ref) and
[`_METRICS`](@ref).

`vbundle` must be a registered [`VBundle`](@ref) with `isref == true`
(the bundle named in [`@def_manifold`](@ref) or [`@def_vbundle`](@ref)).

`@def_metric` is a specialised shortcut (not a call to [`@def_tensor`](@ref))
that always builds a rank-2 fully covariant symmetric metric:

- Slots: `[dual(vbundle), dual(vbundle)]` (both covariant).
- `symmetries=[symmetric(2)]` — fixed, no user override.
- `print_as` set to `string(name)`; `metric` set to `name` (self-referential).
- `_METRICS[name] = vbundle` (vbundle of reference).
- No keyword arguments.

At **expression** time you still write `g[-a1, -a2]` using coordinate indices;
only **definition** uses `@def_metric g tangentM`.

# Examples
~~~julia
@def_manifold M 4 [a1, a2, a3, a4] [A1, A2, A3, A4]

@def_metric g tangentM    # Riemannian metric on M
@def_metric η tangentM    # second metric, same vbundle of reference
~~~
"""
macro def_metric(name, vbundle_expr)
    name isa Symbol ||
        error("@def_metric: first argument must be a symbol, got: $name")
    vbundle_expr isa Symbol ||
        error("@def_metric: second argument must be a vbundle symbol, got: $vbundle_expr")

    vbundle_sym  = QuoteNode(vbundle_expr)
    name_sym     = QuoteNode(name)
    print_as_str = string(name)

    quote
        haskey(_VBUNDLES, $(vbundle_sym)) ||
            error(
                "@def_metric: vbundle $($(vbundle_sym)) is not registered. " *
                "Call @def_manifold or @def_vbundle first."
            )

        local _vb = _VBUNDLES[$(vbundle_sym)]
        getfield(_vb, :isref) ||
            error(
                "@def_metric: vbundle $($(vbundle_sym)) is not a vbundle of reference " *
                "(isref must be true). Pass the ref vbundle, e.g. tangentM not cotangentM."
            )

        local _manifold_sym = getfield(_vb, :manifold)
        local _covariant = getfield(_vb, :dual)

        # ── Guard: warn if redefining ─────────────────────────────────
        if haskey(_TENSORS, $(name_sym))
            @warn "Metric $($(name_sym)) is already defined. Redefining."
        end
        if haskey(_METRICS, $(name_sym))
            delete!(_METRICS, $(name_sym))
        end

        # ── Build metric tensor ────────────────────────────────────────
        local _slots = [_covariant, _covariant]
        local _syms = [symmetric(2)]

        local _tensor_id = _next_tensor_id()
        local _T = Tensor(
            _manifold_sym,
            _slots,
            _syms,
            false,
            Any[],
            $(QuoteNode(print_as_str)),
            $(name_sym),
            _tensor_id,
        )

        # ── Register ─────────────────────────────────────────────────
        _TENSORS[$(name_sym)] = _T
        _METRICS[$(name_sym)] = $(vbundle_sym)
        $(esc(name)) = _T

        println(
            "Defined metric $($(name_sym)) on vbundle of reference $($(vbundle_sym)) " *
            "(manifold :$(_manifold_sym))"
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
    metrics_of_vbundle(vb::Symbol) -> Vector{Symbol}

Return metric names registered on vbundle of reference `vb`, sorted by name.

Used internally by [`@def_tensor`](@ref) when `metric=` is omitted.
"""
function metrics_of_vbundle(vb::Symbol)
    sort([k for (k, v) in _METRICS if v == vb])
end

"""
    metrics_of_manifold(M::Manifold) -> Vector{Symbol}

Return the names of all metrics whose vbundle of reference lies over manifold
`M`, sorted by name. Returns an empty vector if none are defined.
"""
function metrics_of_manifold(M::Manifold)
    sort([
        k for (k, vb_ref) in _METRICS
        if haskey(_VBUNDLES, vb_ref) && _VBUNDLES[vb_ref].manifold == M.name
    ])
end

"""
    is_metric(x) -> Bool

Return `true` if `x` is a [`Tensor`](@ref) instance that is also registered
as a metric in [`_METRICS`](@ref).
"""
function is_metric(x)
    x isa Tensor || return false
    for (sym, t) in _TENSORS
        t === x && return haskey(_METRICS, sym)
    end
    return false
end

"""
    list_metrics() -> Vector{Symbol}

Return the names of all currently registered metrics (across all vbundles).

    @def_metric g tangentM
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
    vb_ref = ""
    for (sym, t) in _TENSORS
        if t === g && haskey(_METRICS, sym)
            vb_ref = string(_METRICS[sym])
            break
        end
    end
    println("Metric:    $(g.print_as)")
    println("  Manifold: $(g.manifold)")
    vb_ref != "" && println("  Vbundle of reference: $vb_ref")
    println("  Slots:    [$(join(g.slots, ", "))]")
    println("  Symmetry: $(g.symmetries)")
end


# =========================================
# 4.  show methods
# =========================================

function _metric_show_line(io::IO, g::Tensor)
    slot_chars = map(g.slots) do vb
        haskey(_VBUNDLES, vb) ? (_VBUNDLES[vb].isref ? "↑" : "↓") : "?"
    end
    print(io,
        "Metric $(g.print_as)[$(join(slot_chars, ""))] on $(g.manifold)"
    )
end


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
export is_metric, list_metrics, metric_info, show_metrics
export metrics_of_manifold, metrics_of_vbundle
