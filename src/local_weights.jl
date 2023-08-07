function col_intersect( x::SubArray{<:Any,<:Any,<:SparseMatrixCSC}, y::SubArray{<:Any,<:Any,<:SparseMatrixCSC}, dx::Td, dy::Td ) where {Td<:Real}

  cap = x' * y
  return cap / (dx + dy - cap)

end

"""
    local_weights!(C::SparseMatrixCSC, A::SparseMatrixCSC)

Compute local weights in matrix C from unweighted matrix A.
"""
function local_weights!(C::SparseMatrixCSC, A::SparseMatrixCSC)

  (m,n) = size(A)
  rows  = rowvals(A)
  @assert all( x -> x == 1, nonzeros(A) )

  # output
  vals  = nonzeros(C)

  @assert m == n

  d = vec( sum(A; dims=1) )

  @inbounds for j = 1:n
    for k in nzrange(A, j)
      i = rows[k]
      i == j && continue
      x = @view A[:,j]
      y = @view A[:,i]
      vals[k] = col_intersect(x, y, d[i], d[j] ) + eps()
    end
  end

  C

end

"""
    local_weights(A::SparseMatrixCSC)

Compute local weights from unweighted matrix A.
"""
local_weights(A::SparseMatrixCSC) = local_weights!( similar(A, Float64), A )