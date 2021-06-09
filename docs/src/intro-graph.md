
# Graph embedding

```@example laxis
using CairoMakie, Colors

f = Figure()

cmap = distinguishable_colors(5, [RGB(1,1,1), RGB(0,0,0)], dropseed=true);

ax = Axis(f[2, 1], title = "2D MNIST")

scatter!( ax, rand(5), rand(5), color = 1:5, colormap = cmap,
          markersize = 15 )
          
ax.aspect = DataAspect()

lgnd_elem = [MarkerElement(color = cmap[i+1],
                           marker = :circle,
                           markersize = 15,
                           strokecolor = :black) for i = 0:4];

Legend(f[1, 1], lgnd_elem, string.( 0:4 ),
    orientation = :horizontal, 
    tellwidth = false, 
    tellheight = true);


f
```

