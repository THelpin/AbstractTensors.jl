
using AbstractTensors
using Documenter
 
DocMeta.setdocmeta!(
    AbstractTensors,
    :DocTestSetup,
    :(using AbstractTensors);
    recursive = true,
)
 
makedocs(;
    modules  = [AbstractTensors],
    authors  = "THelpin <thomas1.helpin@gmail.com> and contributors",
    sitename = "AbstractTensors.jl",
    format   = Documenter.HTML(;
        canonical  = "https://THelpin.github.io/AbstractTensors.jl",
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
    repo       = "github.com/THelpin/AbstractTensors.jl",
    devbranch  = "main",
)
 
