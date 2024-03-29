using SGtSNEpi
using Test
using Graphs
using SparseArrays
using Makie
using Colors

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
          @test typeof( SGtSNEpi.neighbor_recall( X, Y; k = 10, resolution = (800,600) ) ) == Figure
        end

      end

    end

  end

  using FLANN

  @testset "simple, small examples" begin
    
    A = sparse( [1.0 0.0;0.0 1.0] )

    SGtSNEpi.sgtsne_lambda_equalization(sparse(A),3.0)
    SGtSNEpi.perplexity_equalization(sparse(A),3.0)
    SGtSNEpi.sgtsne_lambda_equalization(sparse(A),1e18)
    
  end

  @testset "$knn_type knn" for knn_type ∈ [:flann]

    @testset "profile $profile" for profile ∈ [true, false]

      @testset "d = $d" for d ∈ 1:3

        n = 5000

        X = rand( n, 50 )

        A = pointcloud2graph( X; knn_type )
        
        SGtSNEpi.sgtsne_lambda_equalization( A, 1  );
        SGtSNEpi.sgtsne_lambda_equalization( A, 10 );
        SGtSNEpi.sgtsne_lambda_equalization( A, 1000000 );

        W = pointcloud2graph( X, 5, 10; knn_type, rescale_type = :lambda )
        W = pointcloud2graph( X, 10000, 4; knn_type, rescale_type = :lambda )

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

    cmap = distinguishable_colors(
      maximum(L) - minimum(L) + 1,
      [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
    clr_in = RGBA(cmap[1], 0.2)

    show_embedding( Y; clr_in )

    @test typeof( show_embedding( Y ) ) == Figure
    @test typeof( show_embedding( Y, L ) ) == Figure
    @test typeof( show_embedding( Y, L ; A = A ) ) == Figure
    
  end

  @testset "make sure errors are thrown" begin

    n = 5000

    X = rand( n, 50 )

    @test_throws Exception pointcloud2graph( X; knn_type = :other )
    @test_throws Exception pointcloud2graph( X; rescale_type = :other )

  end

  @testset "graph" begin
    G = watts_strogatz( 1500, 5, 0.05 )
    Y = sgtsnepi( G; d = 2 );
    @test size( Y ) == (1500, 2)
  end

end
