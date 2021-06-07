module SGtSNEpi

# Dependent package
using LinearAlgebra, LightGraphs, SparseArrays, Libdl

# export
export sgtsnepi

# SG-t-SNE C library
libsgtsnepi = C_NULL

# Basic wrappers for sgtsnepi
include( "sgtsne.jl" )
include( "qq.jl" )

# Initialization
function __init__()

  global libsgtsnepi = Libdl.dlopen(
    find_library( "libsgtsnepi", ["/usr/local/lib"] )
  )

end


end
