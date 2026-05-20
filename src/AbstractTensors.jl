module AbstractTensors

# Load order is strict:
#   indices.jl          — TensorIndex, flip, is_up/is_down
#   manifolds.jl        — Manifold, VBundle, @def_manifold (calls register_index!)
#   vbundles.jl         — @def_vbundle (calls @def_frame_bundle when basis= given)
#   permutations.jl     — SignedPerm, SlotSymmetry, canonical_rep
#                         (also defines isless for TensorIndex, needs _VBUNDLES)
#   tensors.jl          — Tensor, @def_tensor (needs all of the above)
#   metrics.jl          — @def_metric
#   tensorExpressions.jl — TensorExpression, show methods
#   frames.jl           — Basis, BasisElement, BasisExpansion, @def_frame_bundle,
#                         basis_expansion (needs TensorExpression, _VBUNDLES)

include("types.jl")
include("indices.jl")
include("manifolds.jl")
include("vbundles.jl")
include("permutations.jl")
include("tensors.jl")
include("metrics.jl")
include("tensorExpressions.jl")
include("frames.jl")

end