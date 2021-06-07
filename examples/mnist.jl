# ========== EMBED MNIST ==========

using SGtSNEpi, MLDatasets, ImageFeatures, Random

X, L = MNIST.traindata(Float64);

n = size(X, 3);

# L = L[1:n_sub];
F = zeros( n, 324 );

for img = 1:n
  F[img,:] = create_descriptor( reverse( X[:,:,img]; dims = 2 ),
                                HOG(; cell_size = 7) )
end

# reproducible results
Random.seed!(0);
Y0 = 0.01 * randn( n, 2 );

Y = sgtsnepi(F; Y0 = Y0);


# ========== VISUALIZE EMBEDDING ==========

using CairoMakie, Colors

f = Figure(resolution = (800,800))
ax = Axis(f[2, 1])

cmap = distinguishable_colors(10);

sc = scatter!(ax, Y[:,1], Y[:,2], color = L, colormap = cmap,
              markersize = 3);
ax.aspect = DataAspect();

lgnd_elem = [MarkerElement(color = cmap[i+1],
                           marker = :circle,
                           markersize = 15,
                           strokecolor = :black) for i = 0:9];

Legend(f[1, 1], lgnd_elem, string.( 0:9 ),
    orientation = :horizontal, tellwidth = false, tellheight = true);

save("/tmp/plot.png", f, px_per_unit = 1)
