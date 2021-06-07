
# SG-t-SNE-Π: Swift Graph Embedding

We provide a `Julia` interface, i.e., a wrapper to SG-t-SNE-Π, which
is a high-performance software for swift embedding of a large, sparse
graph $G(V,E,P)$ into a $d$-dimensional space ($d = 1,2,3$) on a
shared-memory computer. The algorithm SG-t-SNE and the software
t-SNE-Π were first described in Reference ([hpec](@cite)) and released
on [GitHub](https://github.com/fcdimitr/sgtsnepi) in June
2019 ([joss](@cite)). In fact, the software admits two types of data at
input: (i) a graph or network, described in terms of vertices and
edges (pairwise relationships among vertices), directed or undirected;
(ii) a point-cloud data set, described in terms of feature attributes
in numerical values (coordinates), in a metric space. In the second
case, the point-cloud data are converted into a kNN graph, in which
the data points are vertices, the edges represent k-nearest-neighbor
relationships. In other words, the embedding acts as a non-linear
dimension reduction. The software renders 2D or 3D embedding of the
vertices or the data points. Embedding in 3D allows one to observe
connections or disconnections that are obscured in 2D embedding.

SG-t-SNE improves upon two main precursors, t-SNE by
[tsne](@cite), [bhtsne](@cite) and FI-t-SNE by [fitsne](@cite), in
several aspects. (i) It removes the limitation to regular-degree
graphs with t-SNE; (ii) it enables 3D embedding while following the
basic method used in FI-t-SNE for calculating repulsive forces; (iii)
it surpasses in eﬀiciency, the precursor algorithms and
implementations on modern laptop/desktop computers, without
compromising accuracy. Recently, SG-t-SNE-Π has been compared
favorably to UMAP ([scarf](@cite)). This version makes SG-t-SNE readily
deployable to the Julia ecosystem. More information can be found at
[SG-t-SNE-Π](https://github.com/fcdimitr/sgtsnepi).


## References

```@bibliography
```
