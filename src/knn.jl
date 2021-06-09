function _form_knn_graph( X, u, k = 3*u; knn_type = :exact )

  @info "Forming kNN graph [k = $k | type : $knn_type]"

  X = permutedims( X )

  (n,d) = size( X )

  if knn_type == :exact

    nntree = (d < 7 ) ? KDTree(X) : BruteTree(X)
    idxs, dists = NearestNeighbors.knn(nntree, X, k+1, true)

    idxs  = Int32.( hcat( idxs... ) )
    dists = hcat( dists... ) .^ 2  # expects squared distances
  elseif knn_type == :flann

    FP = FLANNParameters(trees = 16, checks = 300 )
    idxs, dists = FLANN.knn(X, X, k+1, FP)

  else
    error("Unknown kNN type $knn_type")
  end

  P = _perplexity_equalize_c( idxs, dists, u )

end


function _perplexity_equalize_c( I::Matrix{Int32}, D::Matrix{Float64}, u::Number )

  @info "Performing perplexity equalization [u = $u]"

  I .-= 1
  k = size(I,1)-1
  n = size(I,2)

  c_m = ccall( ( :perplexityEqualization, libsgtsnepi ), _c_sparse,
                 ( Ptr{Cint},
                   Ptr{Cdouble},
                   Cint,
                   Cint,
                   Cdouble ),
                 I, D,
                 n, k, u )

  r = unsafe_wrap( Array, c_m.row, c_m.nnz ).+1
  c = unsafe_wrap( Array, c_m.col, c_m.n+1 ).+1
  v = unsafe_wrap( Array, c_m.val, c_m.nnz )

  SparseMatrixCSC( c_m.m, c_m.n, c, r, v )
  # Y = permutedims( unsafe_wrap( Array, ptr_y, (d, size(P,1)) ) )

end
