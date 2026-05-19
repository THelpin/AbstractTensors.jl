module AbstractTensors

# Load order is strict:
#   indices.jl    — IndexSymbol, TensorIndex, up/down
#   manifolds.jl  — Manifold, VBundle, @def_manifold (calls register_index!)
#   permutations.jl — SignedPerm, SlotSymmetry, canonical_rep
#                     (also defines isless for TensorIndex, needs _VBUNDLES)
#   tensors.jl    — Tensor, @def_tensor (needs all of the above)

include("types.jl")
include("indices.jl")
include("manifolds.jl")
include("vbundles.jl")
include("permutations.jl")
include("tensors.jl")
include("metrics.jl")
include("tensorExpressions.jl")

end