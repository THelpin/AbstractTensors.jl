# track_allocs.jl
using Profile
using SymbolicTensors
using SymbolicTensors: _MANIFOLDS, _VBUNDLES, _COORDINATE_INDICES, _FRAME_INDICES,
    _TENSORS, _METRICS, _BASES, _FRAME_BUNDLES, _BOUND_BASIS_SYMBOLS,
    _reset_tensor_counter!
import SymbolicTensors: tensor_id

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
    _reset_tensor_counter!()
    return nothing
end
_clear_all_registries!()

# 1. Setup your variables (Copy your setup here)
@def_manifold BM_M 4 [a1, a2, a3, a4, a5, a6, a7, a8] [
    BM_B1, BM_B2, BM_B3, BM_B4
]
@def_metric g tangentBM_M
@def_tensor H [cotangentBM_M, cotangentBM_M]
@def_tensor F [cotangentBM_M, cotangentBM_M]
@def_tensor T [cotangentBM_M, cotangentBM_M]

# Simulated Dict lookup map for bench 4 (anti-pattern baseline).
tensor_ids = Dict{Any, Int}(
    g => tensor_id(g),
    H => tensor_id(H),
    F => tensor_id(F),
    T => tensor_id(T),
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

# 2. WARMUP: Run it once so compilation allocations aren't counted
mid = expr1 * expr2

# 3. CLEAR: Erase all memory tracking data from the warmup
Profile.clear_malloc_data()

# 4. MEASURE: Run the code in a loop to make the allocations massive and obvious
for _ in 1:1000
    mid * expr3
end