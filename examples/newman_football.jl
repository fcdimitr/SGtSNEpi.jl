# ========== EMBED FOOTBALL GRAPH ==========

using SGtSNEpi, MatrixDepot, Random, LinearAlgebra, Colors

md = mdopen("Newman/football")
A = sparse( md.A )
L = Int32.( md.nodevalue )
n = size(A,1)

nc = length( unique(L) )

# reproducible results
Random.seed!(0);
Y0 = 0.01 * randn( n, 2 );

Y = sgtsnepi(A; Y0 = Y0);

# ========== VISUALIZE EMBEDDING ==========

using CairoMakie, Colors

f = Figure(resolution = (800,800))
ax = Axis(f[2, 1])

i,j = findnz( tril(A) )

cmap = distinguishable_colors( nc );

for k = 1:length(i)
  clr = (L[i[k]] == L[j[k]]) ? cmap[ L[i[k]]+1 ] : colorant"#ccbbbbbb"
  lwd = (L[i[k]] == L[j[k]]) ? 2 : 0.5
  lines!( ax, Y[ [i[k],j[k]], 1 ], Y[ [i[k],j[k]], 2 ],
          color = clr, linewidth = lwd )
end

sc = scatter!(ax, Y[:,1], Y[:,2], color = L, colormap = cmap,
              markersize = 14);
ax.aspect = DataAspect();

lgnd_elem = [MarkerElement(color = cmap[i],
                           marker = :circle,
                           markersize = 15,
                           strokecolor = :black) for i = 1:nc];

Legend(f[1, 1], lgnd_elem, string.( sort( unique(L) ) ),
    orientation = :horizontal, tellwidth = false, tellheight = true);

save("/tmp/plot.png", f, px_per_unit = 1)
