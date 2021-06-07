
# SG-t-SNE-Π: Swift Neighbor Embedding of Sparse Stochastic Graphs

We introduce a `Julia` wrapper to SG-t-SNE-Π, a high-performance
software for swift embedding of a large, sparse, stochastic graph
$G(V,E,P)$ into a $d$-dimensional space ($d = 1,2,3$) on a
shared-memory computer. The algorithm SG-t-SNE and the software
t-SNE-Π were first described in Reference [HPEC](@cite). The
algorithm is built upon precursors for embedding a $k$-nearest
neighbor ($k$NN) graph, which is distance-based and regular with
constant degree $k$. In practice, the precursor algorithms are also
limited up to 2D embedding or suffer from overly long latency in 3D
embedding. SG-t-SNE removes the algorithmic restrictions and enables
$d$-dimensional embedding of arbitrary stochastic graphs, including,
but not restricted to, $k$NN graphs. SG-t-SNE-Π expedites the
computation with high-performance functions and materializes 3D
embedding in shorter time than 2D embedding with any precursor
algorithm on modern laptop/desktop computers. More infomration on the
core functionality can be found at the
[SG-t-SNE-Π](https://github.com/fcdimitr/sgtsnepi) `C/C++` repository
(see [JOSS](@cite)).

## References

```@bibliography
```
