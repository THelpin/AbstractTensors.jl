module SymbolicTensors

# Load order is strict:
#   indices.jl          — AbstractIndex, CoordinateIndex, BasisIndex, flip, is_up/is_down
#   manifolds.jl        — Manifold, VBundle, @def_manifold (calls register_coordinate/basis_index!)
#   vbundles.jl         — @def_vbundle (registers BasisIndex)
#   permutations.jl     — SignedPerm, SlotSymmetry, canonical_rep
#                         (also defines isless for AbstractIndex, needs _VBUNDLES)
#   tensors.jl          — Tensor, @def_tensor (needs all of the above)
#   metrics.jl          — @def_metric
#   tensorExpressions.jl — TensorExpression, show methods
#   frames.jl           — Basis (category :coordinate|:frame), BasisElement,
#                         BasisExpansion, ExpansionStyle (Coordinate|Frame),
#                         basis_expansion(T[, style]) — canonical indices only

include("types.jl")
include("indices.jl")
include("manifolds.jl")
include("vbundles.jl")
include("permutations.jl")
include("tensors.jl")
include("metrics.jl")
include("tensorExpressions.jl")
include("frames.jl")
include("show.jl")

"""
    show_registry()

Print a summary of all module-level registries (for REPL debugging).
"""
function show_registry()
    println("=== SymbolicTensors registries ===")
    println("_MANIFOLDS:           ", sort(collect(keys(_MANIFOLDS))))
    println("_VBUNDLES:            ", sort(collect(keys(_VBUNDLES))))
    println("_COORDINATE_INDICES:  ", sort(collect(keys(_COORDINATE_INDICES))))
    println("_BASIS_INDICES:       ", sort(collect(keys(_BASIS_INDICES))))
    println("_TENSORS:              ", sort(collect(keys(_TENSORS))))
    println("_METRICS:             ", sort(collect(keys(_METRICS))))
    println("_BASES keys:          ", sort(collect(keys(_BASES))))
    println("_FRAME_BUNDLES:       ", sort(collect(keys(_FRAME_BUNDLES))))
    return nothing
end

export show_registry

end