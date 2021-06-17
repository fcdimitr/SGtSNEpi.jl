# architecture specifications
@doc raw"""
    l3_cache_size()

Return the size of L3 cache in bytes.
If it fails, the return value is 25 MiB

"""
function l3_cache_size()
  l3 = try    # try to get cache size
    sum( Hwloc.l3cache_sizes() )
  catch  # if failing, assume 25MB
    25 * 2^20
  end
end

@doc raw"""
    num_threads()

Return the number of maximum threads available on the current CPU.
If it fails, the return value is 0.

"""
function num_threads()
  np = try    # try to get number of virtual threads
    Hwloc.num_virtual_cores()
  catch       # otherwise return 0 (let Cilk decide)
    0
  end
end


@doc raw"""
    get_parallelism_strategy_threshold(d::Integer, np::Integer)

Return the grid size limit (per dimension) after which scheme-2
(blue-red) parallelism is used. For smaller size, scheme-1 (thread-local
buffers) parallelism is used.

The decision making is based on being able to avoid eviction of the
thread-local buffers from the L3 cache.

If it fails, the return value is $10^{6/d}$.

"""
function get_parallelism_strategy_threshold(d::Integer, np::Integer)

  l3 = l3_cache_size()
  sd = sizeof( Float64 )

  n_k = if np == 0
    round( (1e6)^(1/d) )
  else
    floor( ( l3 / (sd * np * (d+1)) )^(1/d) )
  end

  Int( n_k )

end
