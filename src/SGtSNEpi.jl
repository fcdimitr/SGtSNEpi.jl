module SGtSNEpi

# Dependent package
using sgtsnepi_jll
using LinearAlgebra, Graphs, SparseArrays, Libdl
using NearestNeighbors
using Colors, LinearAlgebra
using Requires

# export
export sgtsnepi, pointcloud2graph, show_embedding


# C struct to hold sparse matrix
struct _c_sparse
  m::Cint;        # Number of rows
  n::Cint;        # Number of columns
  nnz::Cint;      # Number of nonzero elements

  row::Ptr{Cint};    # Rows indices (NNZ length)
  col::Ptr{Cint};    # Columns offset (N+1 length)
  val::Ptr{Cdouble}; # Values (NNZ length)
end

USING_FLANN = false

# Basic wrappers for sgtsnepi
include( "util.jl" )
include( "knn.jl" )
include( "sgtsne.jl" )
include( "arch_spec.jl" )
include( "local_weights.jl" )

# Initialization
function __init__()
  @require Makie="ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" @eval include("vis.jl")
  @require FLANN="4ef67f76-e0de-5105-ac01-03b6482fb4f8" begin
    global USING_FLANN = true
    using .FLANN
  end
end


end
