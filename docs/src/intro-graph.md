
# Graph embedding


## Prerequisites

You need to install the following packages for this demo

```
using Pkg
Pkg.add(["DataDeps", "MatrixMarket"])
```


We will work with the `optdigits_10NN` graph, from [SuiteSparse Matrix Collection](https://sparse.tamu.edu/ML_Graph/optdigits_10NN). First, we will define a data dependency, to download and open the `MTX` file. 


```@setup 2
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
```

```julia
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
```

We will now embed the graph using SG-t-SNE-Π

```@setup 2
using SGtSNEpi, Random

# reproducible results
Random.seed!(0);
Y0 = 0.01 * randn( size(A,1), 2 );

Y = sgtsnepi(A; Y0 = Y0);
```

```julia
using SGtSNEpi, Random

# reproducible results
Random.seed!(0);
Y0 = 0.01 * randn( size(A,1), 2 );

Y = sgtsnepi(A; Y0 = Y0);
```


## Visualization

To reproduce the next steps, we need to install the following packages

```
Pkg.add(["CairoMakie", "Colors", "Makie"])
```

If `Makie` was not installed when `SGtSNEpi` was loaded, you need to
restart `Julia` and repeat the previous steps.

### Show embedding

Finally, we show the embedding, using the provided visualization function

```@example 2
using CairoMakie, Colors, LinearAlgebra

show_embedding( Y, L ; A = A, res = (2000, 2000) )
```


## 3D embedding

SG-t-SNE-Π enables 3D embedding as well

```@setup 2
using Colors

cmap = distinguishable_colors(
           maximum(L) - minimum(L) + 1,
           [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

Random.seed!(0);
Y0 = 0.01 * randn( size(A,1), 3 );

Y = sgtsnepi(A; d = 3, Y0 = Y0, max_iter = 500);

sc = scatter( Y[:,1], Y[:,2], Y[:,3], color = L, colormap = cmap, markersize = 2 )

record(sc, "sgtsnepi-animation.gif", range(0, 1, length = 24*8); framerate = 24) do ang
  rotate_cam!( sc.figure.scene.children[1], 2*π/(24*8), 0, 0 )
end
```

```julia
using Colors

cmap = distinguishable_colors(
           maximum(L) - minimum(L) + 1,
           [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

Random.seed!(0);
Y0 = 0.01 * randn( size(A,1), 3 );

Y = sgtsnepi(A; d = 3, Y0 = Y0, max_iter = 500);

sc = scatter( Y[:,1], Y[:,2], Y[:,3], color = L, colormap = cmap, markersize = 2 )

record(sc, "sgtsnepi-animation.gif", range(0, 1, length = 24*8); framerate = 24) do ang
  rotate_cam!( sc.figure.scene.children[1], 2*π/(24*8), 0, 0 )
end
```

![sgtsnepi animation](sgtsnepi-animation.gif)

## Clean-up

Remove the data by issuing

```@example 2
rm(datadep"optdigits"; recursive=true)
```
