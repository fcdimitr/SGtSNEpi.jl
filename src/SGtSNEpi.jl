module SGtSNEpi

# Dependent package
using LinearAlgebra, LightGraphs, SparseArrays, Libdl
using NearestNeighbors, FLANN

# export
export sgtsnepi

# SG-t-SNE C library
libsgtsnepi = C_NULL

# C struct to hold sparse matrix
struct _c_sparse
  m::Cint;        # Number of rows
  n::Cint;        # Number of columns
  nnz::Cint;      # Number of nonzero elements

  row::Ptr{Cint};    # Rows indices (NNZ length)
  col::Ptr{Cint};    # Columns offset (N+1 length)
  val::Ptr{Cdouble}; # Values (NNZ length)
end


# Basic wrappers for sgtsnepi
include( "knn.jl" )
include( "sgtsne.jl" )
include( "qq.jl" )

# Initialization
function __init__()

  global libsgtsnepi = Libdl.dlopen(
    find_library( "libsgtsnepi", ["/usr/local/lib"] )
  )

end


end
