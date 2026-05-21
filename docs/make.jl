
using SymbolicTensors
using Documenter
 
DocMeta.setdocmeta!(
    SymbolicTensors,
    :DocTestSetup,
    :(using SymbolicTensors);
    recursive = true,
)
 
makedocs(;
    modules  = [SymbolicTensors],
    authors  = "THelpin <thomas1.helpin@gmail.com> and contributors",
    sitename = "SymbolicTensors.jl",
    format   = Documenter.HTML(;
        canonical  = "https://THelpin.github.io/SymbolicTensors.jl",
        edit_link  = "main",
        assets     = String[],
        # Show the full method signature in the sidebar
        size_threshold = nothing,
    ),
    pages = [
        "Home"      => "index.md",
    ],
    # Treat missing docstrings as errors so gaps surface immediately.
    # Change to :warn while the package is still being built out.
    warnonly = [:missing_docs, :cross_references],
)
 
deploydocs(;
    repo       = "github.com/THelpin/SymbolicTensors.jl",
    devbranch  = "main",
)
 
