# Shared geometry for benchmark/benchmark.jl and benchmark/bench_sort.jl.
# Included after `using SymbolicTensors` in the caller.

using SymbolicTensors: _MANIFOLDS, _VBUNDLES, _COORDINATE_INDICES, _FRAME_INDICES,
    _TENSORS, _METRICS, _BASES, _FRAME_BUNDLES, _BOUND_BASIS_SYMBOLS

function _clear_all_registries!()
    empty!(_MANIFOLDS)
    empty!(_VBUNDLES)
    empty!(_COORDINATE_INDICES)
    empty!(_FRAME_INDICES)
    empty!(_TENSORS)
    empty!(_METRICS)
    empty!(_BASES)
    empty!(_FRAME_BUNDLES)
    empty!(_BOUND_BASIS_SYMBOLS)
    return nothing
end

"""
    setup_benchmark_geometry!()

Define manifold BM_M, metric `g`, tensors H/F/T, and index symbols a1–a8.
Returns a named tuple of handles for benchmarks.
"""
function setup_benchmark_geometry!()
    _clear_all_registries!()

    @def_manifold BM_M 4 [a1, a2, a3, a4, a5, a6, a7, a8] [
        BM_B1, BM_B2, BM_B3, BM_B4
    ]
    @def_metric g tangentBM_M
    @def_tensor H [cotangentBM_M, cotangentBM_M]
    @def_tensor F [cotangentBM_M, cotangentBM_M]
    @def_tensor T [cotangentBM_M, cotangentBM_M]

    # Simulated registry_id map (registration order); for bench_sort micro-benchmark.
    tensor_ids = Dict{Any, Int}(
        g => 1,
        H => 2,
        F => 3,
        T => 4,
    )

    components = [
        g[a1, -a2],
        H[a3, -a4],
        F[a5, -a6],
        T[a7, -a8],
    ]

    expr1 = g[a1, -a2] + H[a1, -a2] + F[a1, -a2] + T[a1, -a2]
    expr2 = g[a3, -a4] + H[a3, -a4] + F[a3, -a4] + T[a3, -a4]
    expr3 = g[a5, -a6] + H[a5, -a6] + F[a5, -a6] + T[a5, -a6]

    return (;
        g, H, F, T,
        a1, a2, a3, a4, a5, a6, a7, a8,
        components,
        tensor_ids,
        expr1,
        expr2,
        expr3,
    )
end
