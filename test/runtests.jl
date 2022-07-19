using SGtSNEpi
using Test
using Graphs
using SparseArrays
using Makie

@testset "SGtSNEpi.jl" begin

  @testset "$knn_type knn" for knn_type ∈ [:exact]

    @testset "profile $profile" for profile ∈ [true, false]

      @testset "d = $d" for d ∈ 1:3

        n = 5000

        X = rand( n, 50 )

        A = pointcloud2graph( X; knn_type )

        @test_throws Exception pointcloud2graph( X; knn_type = :flann )

        Y = sgtsnepi( A; d = d,
                      max_iter = 300, early_exag = 150,
                      profile = profile )

        if profile == true
          @test length( Y ) == 3
          @test size( Y[1] ) == (n, d)
          @test size( Y[2] ) == (6, 300)
          @test size( Y[3] ) == (3, 300)
        else
          @test size( Y ) == (n, d)
        end

      end

    end

  end

  using FLANN

  @testset "$knn_type knn" for knn_type ∈ [:flann]

    @testset "profile $profile" for profile ∈ [true, false]

      @testset "d = $d" for d ∈ 1:3

        n = 5000

        X = rand( n, 50 )

        A = pointcloud2graph( X; knn_type )

        Y = sgtsnepi( A; d = d,
                      max_iter = 300, early_exag = 150,
                      profile = profile )

        if profile == true
          @test length( Y ) == 3
          @test size( Y[1] ) == (n, d)
          @test size( Y[2] ) == (6, 300)
          @test size( Y[3] ) == (3, 300)
        else
          @test size( Y ) == (n, d)
        end

      end

    end

  end

  @testset "graph" begin

    @testset "d = $d" for d ∈ 1:3

      @testset "version = $version" for version = [SGtSNEpi.EXACT, SGtSNEpi.NUCONV, SGtSNEpi.NUCONV_BL]

        n = 2000
        A = sprand( n, n, 0.05 )
        A = A + A'

        Y = sgtsnepi( A; d = d, max_iter = 50, early_exag = 25, version )
        @test size( Y ) == (n, d)

        # check isolated nodes
        A[:,1:5] .= 0
        A[1:5,:] .= 0

        Y = sgtsnepi( A; d = d, max_iter = 50, early_exag = 25, version )
        @test size( Y ) == (n, d)

      end

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

  @testset "make sure errors are thrown" begin

    n = 5000

    X = rand( n, 50 )

    @test_throws Exception pointcloud2graph( X; knn_type = :other )

  end


end
