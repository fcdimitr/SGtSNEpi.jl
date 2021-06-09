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

L = Int32.( vec( L ) );  # make sure labels is an integer vector

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

We visualize the $70{,}000$ digits on the 2D embedding space, colored
by their class. For this purpose, we use the routine `vis_embedding`.


```@example 1
using CairoMakie, Colors

vis_sgtsnepi( Y, L; res = (2000, 2000) )

```
