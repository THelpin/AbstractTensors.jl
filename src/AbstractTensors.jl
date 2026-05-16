module AbstractTensors

# Load order is strict — indices.jl must come first.
# manifolds.jl calls register_index! and unregister_index! which are
# defined in indices.jl. Reversing the order causes an UndefVarError
# at load time.
include("indices.jl")
include("manifolds.jl")

# Uncomment as phases are implemented:
# include("metrics.jl")
# include("tensors.jl")
# include("display.jl")

end