using SGtSNEpi
using Documenter
using DocumenterCitations
using CairoMakie
using Makie

bib = CitationBibliography(joinpath(@__DIR__, "sgtsnepi.bib"))

DocMeta.setdocmeta!(SGtSNEpi, :DocTestSetup, :(using SGtSNEpi); recursive=true)

makedocs( bib,
    modules=[SGtSNEpi],
    authors="Nikos Pitsianis <nikos@cs.duke.edu>, Dimitris Floros <fcdimitr@ece.auth.gr>, Alexandros-Stavros Iliopoulos <ailiop@mit.edu>, Xiaobai Sun <xiaobai@cs.duke.edu>",
    repo="https://github.com/fcdimitr/SGtSNEpi.jl/blob/{commit}{path}#{line}",
    sitename="SGtSNEpi.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://fcdimitr.github.io/SGtSNEpi.jl",
        assets=String[],
        edit_link="master",
        sidebar_sitename=false,
    ),
    doctest = true,
    pages=[
      "Overview" => "index.md",
      "Getting started"  => [
        "Point-cloud data embedding" => "intro-point-cloud.md",
        "Graph embedding" => "intro-graph.md",
      ],
      "API (Advanced)" => "API.md",
    ],
)

deploydocs(;
    repo="github.com/fcdimitr/SGtSNEpi.jl",
    devbranch="master",
    push_preview=true
)



# ========== LaTeX documentation (.tex files)

# using DocumenterLaTeX

# makedocs( bib,
#     modules=[SGtSNEpi],
#     authors="Nikos Pitsianis <nikos@cs.duke.edu>, Dimitris Floros <fcdimitr@ece.auth.gr>, Alexandros-Stavros Iliopoulos <ailiop@mit.edu>, Xiaobai Sun <xiaobai@cs.duke.edu>",
#     repo="https://github.com/fcdimitr/SGtSNEpi.jl/blob/{commit}{path}#{line}",
#     sitename="SGtSNEpi.jl",
#     format = LaTeX(platform = "none"),
#     doctest = false,
#     pages=[
#       "Home" => "index.md",
#       "API" => "API.md",
#     ],
# )
