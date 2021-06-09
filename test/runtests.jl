using SGtSNEpi
using Test
using LightGraphs
using SparseArrays
using Makie

@testset "SGtSNEpi.jl" begin

  @testset "$type knn" for type ∈ [:exact, :flann]

    @testset "d = $d" for d ∈ 1:3

      n = 5000

      X = rand( n, 50 )

      Y = sgtsnepi( X; d = d, knn_type = type, max_iter = 300, early_exag = 150 )

      @test size( Y ) == (n, d)

    end

  end


  @testset "graph" begin

    @testset "d = $d" for d ∈ 1:3

      n = 5000
      A = sprand( n, n, 0.05 )
      A = A + A'
      G = Graph( A )

      Y = sgtsnepi( A; d = d, max_iter = 300, early_exag = 150 )
      @test size( Y ) == (n, d)

      Y = sgtsnepi( G; d = d, max_iter = 300, early_exag = 150 )
      @test size( Y ) == (n, d)

    end

  end

  @testset "qq" begin

    @testset "d = $d" for d ∈ 1:3

      n = 5000
      X = repeat(1:n,1,d) / n;

      Fg,zg = SGtSNEpi.qq( X; type = "exact"  )
      F ,z  = SGtSNEpi.qq( X; type = "interp" )

      @test isapprox(z, zg; rtol = 1e-3)

    end

  end

  @testset "vis" begin

    n = 5000

    A = sprand( n, n, 0.05 )
    A = A + A'
    L = rand( 1:10, n )

    Y = sgtsnepi( A; max_iter = 300, early_exag = 150 )

    f = show_embedding( Y )

    @test typeof( show_embedding( Y ) ) == Figure
    @test typeof( show_embedding( Y, L ) ) == Figure
    @test typeof( show_embedding( Y, L ; A = A ) ) == Figure

  end

end
