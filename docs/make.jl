using AbstractTensors
using Documenter

DocMeta.setdocmeta!(AbstractTensors, :DocTestSetup, :(using AbstractTensors); recursive=true)

makedocs(;
    modules=[AbstractTensors],
    authors="THelpin <thomas1.helpin@gmail.com> and contributors",
    sitename="AbstractTensors.jl",
    format=Documenter.HTML(;
        canonical="https://Thomas.github.io/AbstractTensors.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Thomas/AbstractTensors.jl",
    devbranch="main",
)
