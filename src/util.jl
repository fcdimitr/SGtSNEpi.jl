@doc raw"""
    sgtsne_lambda_equalization(D,λ; maxIter = 50)

Binary search for the scales of column-wise conditional probabilities
from exp(-D) to exp(-D/σ²)/z equalized by λ.

# Inputs

- `D`: ``N \times N`` sparse matrix of "distance square" (column-wise
  conditional, local distances)

- `λ`: The equalization parameter

# Outputs

- `P`: The column-wise conditional probability matrix

# Authors

Xiaobai Sun (MATLAB prototype on May 12, 2019)
Dimitris Floros (translation to Julia)

"""
function sgtsne_lambda_equalization(D::SparseMatrixCSC,λ::Number;
                                    maxIter = 50, tolBinary = 1e-5)

  #############################################################################
  #                          private helper functions                         #
  #############################################################################

  function colsum(D, j, σ = 1.0)
    Dmin  = floatmin(Float64)        # minimum possible value (double prec.)
    vals  = nonzeros(D)              # vector of values
    sum_j = 0;                       # initialize accumulator

    @inbounds for i in nzrange(D, j) # loop over nonzero elements in CSC column j

      sum_j += exp( -vals[i] * σ )       # accumulate nonzero elements

    end

    sum_j = max( sum_j, Dmin )       # make sure value not explicit 0
  end

  function colupdate!(D, j, σ)
    vals  = nonzeros(D)              # vector of values

    @inbounds for i in nzrange(D, j) # loop over nonzero elements in CSC column j

      vals[i] = exp( -vals[i] * σ )  # update element

    end

  end


  #############################################################################
  #                 parameter setting & memory pre-allocations                #
  #############################################################################

  n = size(D, 1)
  condP  = copy( D )

  iDiff  = zeros( n )
  iCount = zeros( n )
  iTval  = zeros( n )
  σ²     = ones(  n );

  #############################################################################
  #                       pre-calculate average entropy                       #
  #############################################################################

  rows = rowvals(D)


  @inbounds for j = 1:n              # loop over all columns of D

    sum_j = colsum( D, j )

    iTval[j] = sum_j - λ             # difference from λ

  end

  #############################################################################
  #                         bisection search for σ²                           #
  #############################################################################


  @inbounds for j = 1:n              # loop over all columns of D

    fval = iTval[j]
    lb, ub = -1000, Inf              # lower/upper bounds for search

    iter = 0

    while abs( fval ) > tolBinary  &&  iter < maxIter

      iter += 1

      if fval > 0                    # update lower bound
        lb = σ²[j]
        σ²[j] = isinf(ub) ? 2*lb : 0.5*(lb + ub)
      else                           # update upper bound
        ub = σ²[j]
        σ²[j] = isinf(lb) ? 0.5*ub : 0.5*(lb + ub)
      end

      # ... re-calculate local entropy
      sum_j = colsum( D, j, σ²[j] )
      fval  = sum_j - λ

    end

    # ... post-recording

    iDiff[j]  = fval;
    iCount[j] = iter;

    colupdate!( condP, j, σ²[j] )

  end

  #############################################################################
  #                      display post-information to user                     #
  #############################################################################

  avgIter = ceil( sum( iCount ) / n )
  nc_idx  = sum( abs.( iDiff ) .> tolBinary )

  if nc_idx == 0
    @info "All $n elements converged numerically, avg(#iter) = $avgIter"
  else
    @warn "There are $nc_idx non-convergent elements out of $N"
  end

  n_neg = sum( σ² .< 0 )
  if n_neg > 0
    @warn "There are $n_neg nodes with negative γᵢ; consider decreasing λ"
  end

  condP

end


@doc raw"""
    sgtsne_lambda_equalization(D,λ; maxIter = 50)

Binary search for the scales of column-wise conditional probabilities
from exp(-D) to exp(-D/σ²)/z equalized by λ.

# Inputs

- `D`: ``N \times N`` sparse matrix of "distance square" (column-wise
  conditional, local distances)

- `u`: perplexity, scalar hyper-parameter, tunable

# Outputs

- `P`: The column-wise conditional probability matrix

# Authors

Xiaobai Sun (MATLAB prototype on May 12, 2019)
Dimitris Floros (translation to Julia)

"""
function perplexity_equalization(D::SparseMatrixCSC, u::Number;
                                 maxIter = 50, tolBinary = 1e-5)

  #############################################################################
  #                          private helper functions                         #
  #############################################################################

  function colentropy(D, j, σ = 1.0)
    Dmin  = floatmin(Float64)        # minimum possible value (double prec.)
    vals  = nonzeros(D)              # vector of values
    sum_j = 0.0;                     # initialize accumulator

    @inbounds for i in nzrange(D, j) # loop over nonzero elements in CSC column j

      sum_j += exp( -vals[i] * σ )       # accumulate nonzero elements

    end

    sum_j = max( sum_j, Dmin )       # make sure value not explicit 0

    h_j = log(sum_j)

    @inbounds for i in nzrange(D, j) # loop over nonzero elements in CSC column j

      P_ij = exp( -vals[i] * σ ) / sum_j

      h_j += σ * vals[i] * P_ij       # accumulate nonzero elements

    end

    h_j

  end

  function colupdate!(D, j, σ)
    vals  = nonzeros(D)              # vector of values

    sum_j = 0.0

    @inbounds for i in nzrange(D, j) # loop over nonzero elements in CSC column j

      sum_j += exp( -vals[i] * σ )       # accumulate nonzero elements

    end

    @inbounds for i in nzrange(D, j) # loop over nonzero elements in CSC column j

      vals[i] = exp( -vals[i] * σ ) / sum_j  # update element

    end

  end


  #############################################################################
  #                 parameter setting & memory pre-allocations                #
  #############################################################################

  n = size(D, 1)
  condP  = copy( D )

  σ² = ones(  n );          # initial value for adaptive σ²
  H  = log( u );          # global entropy

  iTval  = zeros( n )
  iDiff  = zeros( n )
  iCount = zeros( n )

  #############################################################################
  #                       pre-calculate average entropy                       #
  #############################################################################

  rows = rowvals(D)


  @inbounds for j = 1:n              # loop over all columns of D

    sum_j = colentropy( D, j )


    iTval[j] = sum_j - H             # difference from λ

  end

  #############################################################################
  #                         bisection search for σ²                           #
  #############################################################################


  @inbounds for j = 1:n              # loop over all columns of D

    fval = iTval[j]
    lb, ub = -Inf, Inf              # lower/upper bounds for search

    iter = 0

    while abs( fval ) > tolBinary  &&  iter < maxIter

      iter += 1

      if fval > 0                    # update lower bound
        lb = σ²[j]
        σ²[j] = isinf(ub) ? 2*lb : 0.5*(lb + ub)
      else                           # update upper bound
        ub = σ²[j]
        σ²[j] = isinf(lb) ? 0.5*ub : 0.5*(lb + ub)
      end

      # ... re-calculate local entropy
      H_j  = colentropy( D, j, σ²[j] )
      fval = H_j - H

    end

    # ... post-recording

    iDiff[j]  = fval;
    iCount[j] = iter;

    colupdate!( condP, j, σ²[j] )

  end

  #############################################################################
  #                      display post-information to user                     #
  #############################################################################

  avgIter = ceil( sum( iCount ) / n )
  nc_idx  = sum( abs.( iDiff ) .> tolBinary )

  if nc_idx == 0
    @info "All $n elements converged numerically, avg(#iter) = $avgIter"
  else
    @warn "There are $nc_idx non-convergent elements out of $N"
  end

  n_neg = sum( σ² .< 0 )
  if n_neg > 0
    @warn "There are $n_neg nodes with negative γᵢ; consider decreasing λ"
  end

  condP

end
