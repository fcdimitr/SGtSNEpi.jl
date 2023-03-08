function _form_knn_graph( X, u, k = 3*u; knn_type = :exact, rescale_type = :perplexity )

  @info "Forming kNN graph [k = $k | type : $knn_type]"

  X = permutedims( X )

  (d,n) = size( X )

  if knn_type == :exact

    nntree = (d < 7 ) ? KDTree(X) : BruteTree(X)
    idxs, dists = NearestNeighbors.knn(nntree, X, k+1, true)

    idxs  = Int32.( hcat( idxs... ) )
    dists = hcat( dists... ) .^ 2  # expects squared distances
  elseif knn_type == :flann && USING_FLANN

    FP = FLANNParameters(trees = 16, checks = 300 )
    idxs, dists = FLANN.knn(X, X, k+1, FP)

  else
    error("Unknown kNN type $knn_type. If you want to use FLANN, issue `using FLANN` before calling this function.")
  end

  D² = sparse( vec(idxs), vec( repeat((1:n)', k+1, 1) ), vec(dists) )
  D² = D² - spdiagm( diag(D²) )
  P  = if rescale_type == :perplexity
    perplexity_equalization( D², u )
  elseif rescale_type == :lambda
    sgtsne_lambda_equalization( D², u )
  else
    error("Unknown rescale type $rescale_type")
  end

  # P = _perplexity_equalize_c( idxs, dists, u )

end
