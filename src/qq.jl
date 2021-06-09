@doc raw"""
    qq( X::DenseMatrix [; type = "exact", h = 1, np = 1 ])

Compute repulsive forces (QQ) of SG-t-SNE, approximatelly (linear
complexity) or exactly with O(n^2) complexity. The input `X` is
of size $n \times d$.

## Optional arguments

- `type="exact"`: either "exact" or "interp"
- "h=1"; the grid box side length *only for `interp` type*
- "np=1": number of processors *only for `interp` type*

# Examples
```jldoctest
julia> X = repeat(1:1000,1,2) / 1000.0;

julia> F,z = SGtSNEpi.qq( X );


julia> @show floor( z );
floor(z) = 800716.0

julia> F,z = SGtSNEpi.qq( X; type = "interp" );


julia> @show floor( z );
floor(z) = 800685.0

```
"""
function qq( X::DenseMatrix ; type = "exact", h = 1, np = 1 )

  if type == "exact"

    _qq_exact_c( X )

  elseif type == "interp"

    _qq_interp_c( X, h, np )

  else

    throw( "Unknonwn type: $type" )

  end

end


function _qq_exact_c( X::DenseMatrix )

  X = X'

  frep = zeros( size(X) )

  zeta = ccall( ( :computeFrepulsive_exact, libsgtsnepi ),
                Cdouble,
                ( Ptr{Cdouble},
                  Ptr{Cdouble},
                  Cint, Cint ),
                frep, X,
                size(X,2), size(X,1) )

  permutedims( frep ), zeta

end

function _qq_interp_c( X::DenseMatrix, h::Real, np::Int )

  X = X'

  frep = zeros( size(X) )
  timers = zeros( 5 )

  zeta = ccall( ( :computeFrepulsive_interp, libsgtsnepi ),
                Cdouble,
                ( Ptr{Cdouble},
                  Ptr{Cdouble},
                  Cint, Cint,
                  Cdouble, Cint, Ptr{Cdouble}),
                frep, X,
                size(X,2), size(X,1),
                h, np, timers )

  permutedims( frep ), zeta

end
