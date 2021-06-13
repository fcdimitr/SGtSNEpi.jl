
@doc raw"""
    sgtsnepi( A::AbstractMatrix )
    sgtsnepi( G::AbstractGraph )

Call SG-t-SNE-Π on the input graph, given as either a sparse adjacency
matrix $A$ or a graph object $G$. Alternatively, the input can be a
point-cloud data set $X$ (coordinates) of size $N \times D$, i.e.,

    sgtsnepi( X::AbstractMatrix )

## Optional arguments

- `d=2`: number of dimensions (embedding space)
- `λ=10`: SG-t-SNE scaling factor

## More options (for experts)

- `max_iter=1000`: number of iterations
- `early_exag=250`: number of early exageration iterations
- `np=0`: number of processors (set to 0 to use all available cores)
- `Y0=nothing`: initial distribution in embedding space (randomly selected if `nothing`)
- `h=0`: grid side length (0 for default values)
- `eta=200.0`: learning parameter
- `bound_box=Inf`: maximum bounding box (rescaling after that)
- `profile=false`: disable/enable profiling. If enabled the function
  return a 3-tuple: `(Y, t, g)`, where `Y` is the embedding
  coordinates, `t` are the execution times per iteration and `g` is
  the grid size per iteration.

## Special options for point-cloud data embedding

- `u=10`: perplexity
- `k=3*u`: number of nearest neighbors (for kNN formation)
- `knn_type=( size(A,1) < 10_000 ) ? :exact : :flann`: Exact or approximate kNN

## Notes

- Isolated are placed randomly on the top-right corner of the
  embedding space

- The function tries to automatically detect whether the input matrix
  represents an adjacency matrix or data coordinates. In ambiquous cases,
  such as a square matrix of data coordinates, the user may
  specify the type using the optional argument `type`
  - `:graph`: the input is an adjacency matrix
  - `:coord`: the input is the data coordinates



# Examples
```jldoctest; filter = [r".*error is.*", r".*seconds.*", r".*Attractive.*"]
julia> using LightGraphs

julia> Y0 = repeat(1:1000,1,2) / 1000.0;

julia> G = circular_ladder_graph( 500 )
{1000, 1500} undirected simple Int64 graph

julia> Y = sgtsnepi( G; Y0 = Y0, np = 4, early_exag = 100, max_iter = 250 );
Number of vertices: 1000
Embedding dimensions: 2
Rescaling parameter λ: 10
Early exag. multiplier α: 12
Maximum iterations: 250
Early exag. iterations: 100
Learning rate: 200
Box side length h: 0.7
Drop edges originating from leaf nodes? 0
Number of processes: 4
1000 out of 1000 nodes already stochastic
m = 1000 | n = 1000 | nnz = 3000
Working with double precision
Iteration 1: error is 96.9204
Iteration 50: error is 84.9181 (50 iterations in 0.039296 seconds)
Iteration 100: error is 4.32754 (50 iterations in 0.038005 seconds)
Iteration 150: error is 2.54655 (50 iterations in 0.066491 seconds)
Iteration 200: error is 1.90124 (50 iterations in 0.159556 seconds)
Iteration 249: error is 1.65057 (50 iterations in 0.213149 seconds)
 --- Time spent in each module ---

 Attractive forces: 0.006199 sec [1.24082%] |  Repulsive forces: 0.49339 sec [98.7592%]
```
"""
sgtsnepi( G::AbstractGraph ; kwargs... ) = sgtsnepi( Float64.( adjacency_matrix(G) ) ; kwargs..., type = :graph )

function sgtsnepi( A::AbstractMatrix ;
                   d = 2, λ = 10,
                   max_iter = 1000, early_exag = 250,
                   Y0 = nothing,
                   profile = false, np = 0,
                   type = nothing,
                   h = 0,
                   u = 10,
                   k = 3*u,
                   eta = 200.0,
                   bound_box = Inf,
                   knn_type = ( size(A,1) < 10_000 ) ? :exact : :flann )

  A = ( isequal( size(A)... ) && type != :coord ) ? A :
    _form_knn_graph( A, u, k ; knn_type = knn_type )

  A = issparse( A ) ? A : sparse( A )

  nnz( diag(A) ) > 0 && @warn "$( nnz( diag(A) ) ) elements have self-loops; setting distances to 0"
  A = A - spdiagm( 0 => diag( A ) )
  dropzeros!( A )

  @assert nnz( diag(A) ) == 0

  n = size( A, 1 )

  Y0 = ( isnothing( Y0 ) ) ? C_NULL : Y0

  Y0 != C_NULL && size( Y0 ) != (n, d) && error( "Incorrect initial distribution size: $(size(Y0))" )

  # transform input matrix to stochastic; isolated nodes are removed, index contains valid IDs
  P, idx = colstoch( A )

  Y = zeros( n, d );

  if (profile)
    Y[idx,:],t,g = _sgtsnepi_profile_c( P, d, max_iter, early_exag, λ; Y0 = Y0, np = np, h = h, bb = bound_box, eta = eta )
    Y = _fix_isolated( Y, idx )
    Y,t,g
  else
    Y[idx,:] = _sgtsnepi_c( P, d, max_iter, early_exag, λ; Y0 = Y0, np = np, h = h, bb = bound_box, eta = eta )
    Y = _fix_isolated( Y, idx )
    Y
  end


end

function _fix_isolated( Y, idx )

  n_isolated = sum(.!idx)
  d = size( Y, 2 )

  if n_isolated == 0
    return Y
  end

  corner = maximum( Y[idx,:]; dims = 1 )

  Y[.!idx,:] .= corner .* ( 1.0 .+ rand( n_isolated, d ) ./ 10.0 )

  Y

end

function colstoch(A)
  idxKeep = vec( sum(A,dims=1) ) .!= 0;

  !all( idxKeep ) && @warn "$( sum( .! idxKeep ) ) isolated nodes; they are placed at (0,0,...)"

  A = A[idxKeep,idxKeep]
  D = spdiagm( 0 => 1 ./ vec( sum(A;dims=1) ) );
  P = A * D;
  P, idxKeep
end

function _sgtsnepi_c( P::SparseMatrixCSC, d::Int, max_iter::Int, early_exag::Int, λ::Real; Y0 = C_NULL, np = 0, h = 0.0, bb = Inf, eta = 200.0 )

  Y0 = (Y0 == C_NULL) ? C_NULL : permutedims( Y0 )

  rows = Int32.( P.rowval .- 1 );
  cols = Int32.( P.colptr .- 1 );
  vals = Float64.( P.nzval );

  if isinf(bb)
    bb = h * ( size(P, 1) ^ (1/d) ) / 2
  end

  @debug bb

  if h == 0.0
    h = C_NULL
  else
    h = Float64.( isempty(size(h)) ? [max_iter+1, h] : h )
    ( length( h ) % 2 ) == 0 || error( "h must have even number of elements" )
    ( h[end-1] >= max_iter ) || error( "last phase should be equal or greater to max_iter" )
  end

  @debug h

  ptr_y = ccall( ( :tsnepi_c, libsgtsnepi ), Ptr{Cdouble},
                 ( Ptr{Ptr{Cdouble}}, Ptr{Cint},
                   Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble},
                   Ptr{Cdouble},
                   Cint,
                   Cint, Cdouble, Cint, Cint,
                   Ptr{Cdouble}, Cdouble, Cdouble,
                   Cint, Cint ),
                 C_NULL, C_NULL,
                 rows, cols, vals,
                 Y0,
                 Int32.( nnz(P) ),
                 d, λ, max_iter, early_exag,
                 h, bb, eta,
                 Int32.( size(P,1) ), np )

  Y = permutedims( unsafe_wrap( Array, ptr_y, (d, size(P,1)) ) )

end


function _sgtsnepi_profile_c( P::SparseMatrixCSC, d::Int, max_iter::Int, early_exag::Int, λ::Real; Y0 = C_NULL, np = 0, h = 0.0, bb = Inf, eta = 200.0 )

  Y0 = (Y0 == C_NULL) ? C_NULL : permutedims( Y0 )

  timers = zeros( Float64, 6, max_iter );
  ptr_timers = Ref{Ptr{Cdouble}}([Ref(timers,i) for i=1:size(timers,1):length(timers)]);

  grid_sizes = zeros( Int32, max_iter );

  rows = Int32.( P.rowval .- 1 );
  cols = Int32.( P.colptr .- 1 );
  vals = Float64.( P.nzval );

  if isinf(bb)
    bb = h * ( size(P, 1) ^ (1/d) ) / 2
  end

  @debug bb

  if h == 0.0
    h = C_NULL
  else
    h = Float64.( isempty(size(h)) ? [max_iter+1, h] : h )
    ( length( h ) % 2 ) == 0 || error( "h must have even number of elements" )
    ( h[end-1] >= max_iter ) || error( "last phase should be equal or greater to max_iter" )
  end

  @debug h

  ptr_y = ccall( ( :tsnepi_c, libsgtsnepi ), Ptr{Cdouble},
                 ( Ptr{Ptr{Cdouble}}, Ptr{Cint},
                   Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble},
                   Ptr{Cdouble},
                   Cint,
                   Cint, Cdouble, Cint, Cint,
                   Ptr{Cdouble}, Cdouble, Cdouble,
                   Cint, Cint ),
                 ptr_timers, grid_sizes,
                 rows, cols, vals,
                 Y0,
                 Int32.( nnz(P) ),
                 d, λ, max_iter, early_exag,
                 h, bb, eta,
                 Int32.( size(P,1) ), np )

  Y = permutedims( unsafe_wrap( Array, ptr_y, (d, size(P,1)) ) )

  Y, timers, grid_sizes

end


