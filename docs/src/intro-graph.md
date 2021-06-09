
# Graph embedding


## Prerequisites

You need to install the following packages for this demo

```
using Pkg
Pkg.add(["DataDeps", "MatrixMarket"])
```


We will work with the `optdigits_10NN` graph, from [SuiteSparse Matrix Collection](https://sparse.tamu.edu/ML_Graph/optdigits_10NN). First, we will define a data dependency, to easily download and open the `MTX` file. 


```@example 2
using DataDeps, MatrixMarket

register(DataDep("optdigits",
               """
       ML_Graph: adjacency matrices from machine learning datasets, Olaf
       Schenk.  D.  Pasadakis,  C.  L.  Alappat,  O.  Schenk,  and  G.
       Wellein, "K-way p-spectral clustering on Grassmann manifolds," 2020.
       https://arxiv.org/abs/2008.13210

       Graph: optdigits_10NN Classes: 10

       See https://sparse.tamu.edu/ML_Graph/optdigits_10NN
               """,
               "https://suitesparse-collection-website.herokuapp.com/MM/ML_Graph/optdigits_10NN.tar.gz",
               "336949bf9a2ac3a0643a8d7a2217792f52e0fdbec54e1b870079f257f200abfc",
               post_fetch_method = unpack
))

A = MatrixMarket.mmread( datadep"optdigits/optdigits_10NN/optdigits_10NN.mtx" );
L = vec( MatrixMarket.mmread( datadep"optdigits/optdigits_10NN/optdigits_10NN_label.mtx" ) );
L = Int.( L );

nothing # hide
```

We will now embed the graph using SG-t-SNE-Π

```@example 2
using SGtSNEpi, Random

# reproducible results
Random.seed!(0);
Y0 = 0.01 * randn( size(A,1), 2 );

Y = sgtsnepi(A; Y0 = Y0);
nothing # hide
```

## Visualization

To reproduce the next steps, you need to install the following packages

```
Pkg.add(["CairoMakie", "Colors"])
```


### Helper function

```@example 2
using CairoMakie, Colors, LinearAlgebra, SparseArrays

function show_embedding(
  Y, L::Vector{Int} = zeros( Int, size(Y,1) );
  cmap = distinguishable_colors(
    maximum(L) - minimum(L) + 1,
    [RGB(1,1,1), RGB(0,0,0)], dropseed=true),
  A = nothing,
  res  = (800, 800),
  lwd_in  = 0.5,
  lwd_out = 0.3,
  edge_alpha = 0.2,
  clr_out = colorant"#aabbbbbb",
  mrk_size = 4,
  size_label = 24 )


  function _plot_lines!( ax, Y, i, j, idx, color, lwd )
    ii = vec( [reshape( Y[ vcat( i[idx], j[idx] ), 1 ], (Int32.(sum(idx)), 2) ) NaN*zeros(sum(idx))]' )
    jj = vec( [reshape( Y[ vcat( i[idx], j[idx] ), 2 ], (Int32.(sum(idx)), 2) ) NaN*zeros(sum(idx))]' )

    lines!( ax, ii, jj, color = color, linewidth = lwd )

  end


  n = size( Y, 1 )

  L .= L .- minimum( L )
  L_u = sort( unique( L ) )
  nc  = length( L_u )

  f = Figure(resolution = res)
  ax = (nc>1) ? Axis(f[2, 1]) : Axis(f[1, 1])


  if !isnothing( A )
    i,j = findnz( tril(A) )
    for kk ∈ L_u
      idx_inner = map( (x,y) -> x == y && x == kk, L[i], L[j] )
      _plot_lines!( ax, Y, i, j, idx_inner, RGBA(cmap[kk+1], edge_alpha), lwd_in )
    end
    idx_cross = map( (x,y) -> x != y, L[i], L[j] )
    _plot_lines!( ax, Y, i, j, idx_cross, colorant"#aabbbbbb", lwd_out )
  end

  scatter!(ax, Y[:,1], Y[:,2], color = L, colormap = cmap,
           markersize = mrk_size);
  ax.aspect = DataAspect();

  lgnd_elem = [MarkerElement(color = cmap[i+1],
                             marker = :circle,
                             markersize = 20,
                             strokecolor = :black) for i = L_u];


  if nc > 1
    Legend(f[1, 1], lgnd_elem, string.( sort( unique(L) ) ),
           orientation = :horizontal, tellwidth = false, tellheight = true,
           labelsize = size_label);
  end

  f

end

```

### Show embedding

Finally, we show the embedding, using the custom function defined at
the beginning of the file

```@example 2
using CairoMakie, Colors, LinearAlgebra

show_embedding( Y, L ; A = A, res = (2000, 2000) )
```


## Clean-up

Remove the data by issuing

```@example 2
rm(datadep"optdigits"; recursive=true)
```
