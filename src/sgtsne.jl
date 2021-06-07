
@doc raw"""
    sgtsnepi( A [; d = 2, max_iter = 1000, early_exag = 250, λ = 10, np = 0, profile = false, Y0 = nothing ])
    sgtsnepi( G [; d = 2, max_iter = 1000, early_exag = 250, λ = 10, np = 0, profile = false, Y0 = nothing ])

Call SG-t-SNE-Π on the input graph, given as either a sparse adjacency
matrix or a graph object.

## Optional arguments

- `d=2`: number of dimensions (embedding space)
- `max_iter=1000`: number of iterations
- `early_exag=250`: number of early exageration iterations
- `λ=10`: SG-t-SNE scaling factor
- `np=0`: number of processos (set 0 to automatically detect)
- `Y0=nothing`: initial distribution in embedding space (randomly selected if `nothing`)
- `profile=false`: disable/enable profiling. If enabled the function
  return a 3-tuple: `(Y, t, g)`, where `Y` is the embedding
  coordinates, `t` are the execution times per iteration and `g` is
  the grid size per iteration.

## Notes

Isolated are embedded at (0,0,...)

# Examples
```jldoctest; filter = [r".*seconds.*", r".*Attractive.*"]
julia> using LightGraphs

julia> Y0 = repeat(1:1000,1,2) / 1000.0;

julia> G = circular_ladder_graph( 500 )
{1000, 1500} undirected simple Int64 graph

julia> Y = sgtsnepi( G; Y0 = Y0, np = 4, early_exag = 100, max_iter = 250 );
input nnz: 3000
Number of vertices: 1000
Embedding dimensions: 2
Rescaling parameter λ: 10
Early exag. multiplier α: 12
Maximum iterations: 250
Early exag. iterations: 100
Box side length h: 0.7
Drop edges originating from leaf nodes? 0
Number of processes: 4
m = 1000 | n = 1000 | nnz = 3000
1000 out of 1000 nodes already stochastic
m = 1000 | n = 1000 | nnz = 3000
m = 1000 | n = 1000 | nnz = 3000
m = 1000 | n = 1000 | nnz = 3000
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
sgtsnepi( G::AbstractGraph ; kwargs... ) = sgtsnepi( Float64.( adjacency_matrix(G) ) ; kwargs... )

function sgtsnepi( A::SparseMatrixCSC ; d = 2, max_iter = 1000, early_exag = 250, λ = 10, profile = false, Y0 = nothing, np = 0 )

  nnz( diag(A) ) > 0 && @warn "$( nnz( diag(A) ) ) elements have self-loops; setting distances to 0"
  A = A - spdiagm( 0 => diag( A ) )

  @assert nnz( diag(A) ) == 0

  n = size( A, 1 )

  Y0 = ( isnothing( Y0 ) ) ? C_NULL : Y0

  Y0 != C_NULL && size( Y0 ) != (n, d) && error( "Incorrect initial distribution size: $(size(Y0))" )

  # transform input matrix to stochastic; isolated nodes are removed, index contains valid IDs
  P, idx = colstoch( A )

  Y = zeros( n, d );

  if (profile)
    Y[idx,:],t,g = _sgtsnepi_profile_c( P, d, max_iter, early_exag, λ; Y0 = Y0, np = np )
    Y,t,g
  else
    Y[idx,:] = _sgtsnepi_c( P, d, max_iter, early_exag, λ; Y0 = Y0, np = np )
    Y
  end

end

function colstoch(A)
  idxKeep = .! ( vec( sum(A,dims=1) ) .== 0 );
  A = A[idxKeep,idxKeep]
  D = spdiagm( 0 => 1 ./ vec( sum(A;dims=1) ) );
  P = A * D;
  P, idxKeep
end

function _sgtsnepi_c( P::SparseMatrixCSC, d::Int, max_iter::Int, early_exag::Int, λ::Real; Y0 = C_NULL, np = 0 )

  Y0 = (Y0 == C_NULL) ? C_NULL : permutedims( Y0 )

  rows = Int32.( P.rowval .- 1 );
  cols = Int32.( P.colptr .- 1 );
  vals = Float64.( P.nzval );

  ptr_y = ccall( dlsym( libsgtsnepi, :tsnepi_c ), Ptr{Cdouble},
                 ( Ptr{Ptr{Cdouble}}, Ptr{Cint},
                   Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble},
                   Ptr{Cdouble},
                   Cint,
                   Cint, Cdouble, Cint, Cint,
                   Cint, Cint ),
                 C_NULL, C_NULL,
                 rows, cols, vals,
                 Y0,
                 Int32.( nnz(P) ),
                 d, λ, max_iter, early_exag,
                 Int32.( size(P,1) ), np )

  Y = permutedims( unsafe_wrap( Array, ptr_y, (d, size(P,1)) ) )

end


function _sgtsnepi_profile_c( P::SparseMatrixCSC, d::Int, max_iter::Int, early_exag::Int, λ::Real; Y0 = C_NULL, np = 0 )

  Y0 = (Y0 == C_NULL) ? C_NULL : permutedims( Y0 )

  timers = zeros( Float64, 6, max_iter );
  ptr_timers = Ref{Ptr{Cdouble}}([Ref(timers,i) for i=1:size(timers,1):length(timers)]);

  grid_sizes = zeros( Int32, max_iter );

  rows = Int32.( P.rowval .- 1 );
  cols = Int32.( P.colptr .- 1 );
  vals = Float64.( P.nzval );

  ptr_y = ccall( dlsym( libsgtsnepi, :tsnepi_c ), Ptr{Cdouble},
                 ( Ptr{Ptr{Cdouble}}, Ptr{Cint},
                   Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble},
                   Ptr{Cdouble},
                   Cint,
                   Cint, Cdouble, Cint, Cint,
                   Cint, Cint ),
                 ptr_timers, grid_sizes,
                 rows, cols, vals,
                 Y0,
                 Int32.( nnz(P) ),
                 d, λ, max_iter, early_exag,
                 Int32.( size(P,1) ), np )

  Y = permutedims( unsafe_wrap( Array, ptr_y, (d, size(P,1)) ) )

  Y, timers, grid_sizes

end


