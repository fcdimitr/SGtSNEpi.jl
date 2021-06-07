using SGtSNEpi
using Documenter

DocMeta.setdocmeta!(SGtSNEpi, :DocTestSetup, :(using SGtSNEpi); recursive=true)

makedocs(;
    modules=[SGtSNEpi],
    authors="Nikos Pitsianis <nikos@cs.duke.edu>, Dimitris Floros <fcdimitr@ece.auth.gr>, Alexandros-Stavros Iliopoulos <ailiop@mit.edu>, Xiaobai Sun <xiaobai@cs.duke.edu>",
    repo="https://github.com/fcdimitr/SGtSNEpi.jl/blob/{commit}{path}#{line}",
    sitename="SGtSNEpi.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://fcdimitr.github.io/SGtSNEpi.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/fcdimitr/SGtSNEpi.jl",
)
