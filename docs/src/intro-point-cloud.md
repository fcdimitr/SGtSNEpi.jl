```@eval
using CairoMakie
CairoMakie.activate!()
```

# Point-cloud data embedding

We provide a case study, using the `MNIST` dataset, to get you started
with SG-t-SNE-Π. We assume you have [Julia](https://julialang.org/)
and `SGtSNEpi` installed already.


## Prerequisites

You need to install the following packages for this demo

```
using Pkg
Pkg.add(["MLDatasets", "ImageFeatures", "Random", "Images"])
```

## MNIST data

The `MNIST` dataset comprises of $60{,}000$ training and $10{,}000$
testing images of handwritten digits. We shall embed the total of
$70{,}000$ handwritten images.

First, we download the dataset

```@example 1
using SGtSNEpi, MLDatasets, ImageFeatures, Random, Images

X, L = MNIST.traindata(Float64);
X = cat( X, MNIST.testdata(Float64)[1] ; dims = 3 );
L = cat( L, MNIST.testdata(Float64)[2] ; dims = 1 );

L = Int.( vec( L ) );  # make sure labels is an integer vector

n = size( X, 3 );

X = permutedims( X, [2, 1, 3] );

nothing; # hide
```

We visualize some of the digits that appear in the data set

```@example 1
mosaicview( Gray.(X[:,:,1:600]), ncol=30, rowmajor=true )
```

We transform the pixel values to Histogram of Oriented Gradients (HOG)
descriptors

```@example 1
F = zeros( n, 324 );

for img = 1:n
  F[img,:] = create_descriptor( X[:,:,img], HOG(; cell_size = 7) )
end
```

We initialize (randomly) the coordinates in the 2D embedding space
(this step is crucial for reproducible results)

```@example 1
Random.seed!(0);
Y0 = 0.01 * randn( n, 2 );
nothing; # hide
```

We use SG-t-SNE-Π to embed the data in a 2D space

```@example 1
Y = sgtsnepi(F; Y0 = Y0);
nothing; # hide
```

## Visualization

To reproduce the next steps, you need to install the following packages
```
Pkg.add(["CairoMakie", "Colors"])
```

### Helper function

```@example 1
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
  ax = (nc>1) ? CairoMakie.Axis(f[2, 1]) : CairoMakie.Axis(f[1, 1])


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

### Visualization

We visualize the $70{,}000$ digits on the 2D embedding space, colored
by their class. For this purpose, we use the routine `vis_embedding`.


```@example 1

show_embedding( Y, L; res = (2000, 2000) )
```
