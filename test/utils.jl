using TensorKrylov, Test, SparseArrays

@testset "Masked products" begin

    n = 5
    d = 4
    matrices    = [ rand(n, n) for _ in 1:d ]
    values      = ones(n, n)
    test_values = ones(n, n)

    for j = 1:n, i = 1:n
        
        values[i, j] = maskprod(matrices, i, j) 

    end

    for s = 1:d

        for j = 1:n, i = 1:n

            test_values[i, j] *= matrices[s][i, j]

        end

    end

    @test all(values .== test_values)

    vec_matrix = [ rand(n, n) ]

    for j = 1:n, i = 1:n

        values[i, j] = maskprod(vec_matrix, i, j)

    end

    test_values .= copy(vec_matrix...)

    @test all(values .== test_values)

end

@testset "Test data updates at each iterative step" begin

    @testset "Update right-hand side" begin

        d  = 2
        n  = 20
        k  = 10
        bs = rand(n)
        b  = [ bs for _ in 1:d ]
        b̃  = [ zeros(n) for _ in eachindex(b) ]
        V  = KronMat{SymInstance}([ rand(n, k) for _ in 1:d ])

        @test test_update_rhs(b̃, V, b, k)
        
    end

    #@testset "Update spectral data"
    
end

@testset "Compressed system solution" begin

    d    = 2
    n    = 4
    rank = 2
    Hs = diagm( log.( ones(n) ) )              # exp(Hs) = I(n)
    H  = KronMat{Instance}([ Hs for _ in 1:d ])
    b  = [ ones(n) for _ in 1:d ]              # exp(Hs) * bs = [1, ..., 1]ᵀ
    y = KruskalTensor{Float64}(ones(rank), [ ones(n, rank) for _ in 1:d ])


    @testset "Matrix exponential vector" begin

        for k = 1:rank

            matrix_exponential_vector!(y, H, b, 1.0, k, MatrixGallery) 

        end

        @test all( y.fmat[s] == ones(n, rank) for s in 1:d )

    end

end

@testset "Compressed residual computations" begin

    @testset "Residual computation in symmetric case" begin

        d    = 3
        n    = 15
        rank = 4

        # Example of linearly dependent factor matrices in CP-decomposition
        Mᵢ  = [ SymTridiagonal(2ones(n), -ones(n-1)) for _ in 1:d ]
        M   = KroneckerMatrix{SymInstance}(Mᵢ)
        mat = rand(n, rank)
        X   = [ rand() .* mat for _ in 1:d ] # Only passes for linearly dependent factor matrices
        #X   = [ rand(n, rank) for _ in 1:d ]
        x   = KruskalTensor{Float64}(ones(rank), X)

        Λ, lowerX, Z = initialize_matrix_products(M, x)

        for s in 1:d

            @test M[s] * X[s] ≈ Z[s]

        end

        solution = exact_solution(M, x)

        @test error_MVnorm(x, Λ, lowerX, Z, solution) < 1e-14


        # Generate right-hand side
        b = [ rand(n) for _ in 1:d ]

        normalize!(b)

        # If factor matrices in CP-decomposition are close to being orthogonal the dot product computation is very ill-conditioned.
        @test error_tensorinnerprod(Z, x, b, solution) < 1e-14

        # Explicit compressed residual norm
        @test error_compressed_residualnorm(b, solution, Λ, lowerX, M, x) < 1e-15

    end

    @testset "Residual computation in nonsymmetric case" begin

        d    = 3
        n    = 15
        rank = 4

        # Example of linearly dependent factor matrices in CP-decomposition
        #M   = KroneckerMatrix{Float64}([ rand(n, n) for _ in 1:d ])
        M   = KroneckerMatrix{NonSymInstance}([ assemble_matrix(n, ConvDiff) for _ in 1:d])
        mat = rand(n, rank)
        X   = [ rand() .* mat for _ in 1:d ] # Only passes for linearly dependent factor matrices
        x   = KruskalTensor{Float64}(ones(rank), X)

        Λ, lowerX, Z = initialize_matrix_products(M, x)

        for s in 1:d

            @test M[s] * X[s] ≈ Z[s]

        end

        solution = exact_solution(M, x)

        @test error_MVnorm(x, Λ, lowerX, Z, solution) < 1e-14


        # Generate right-hand side
        b = [ rand(n) for _ in 1:d ]


        # If factor matrices in CP-decomposition are close to being orthogonal the dot product computation is very ill-conditioned.
        @test error_tensorinnerprod(Z, x, b, solution) < 1e-14

        # Explicit compressed residual norm
        @test error_compressed_residualnorm(b, solution, Λ, lowerX, M, x) < 1e-15

    end


    @testset "Residual computation simulating method scenario" begin

        d = 3
        n = 15
        h = inv(n + 1)

        # Example of linearly dependent factor matrices in CP-decomposition
        Mᵢ  = [ sparse(inv(h^2) .* Tridiagonal(-ones(n-1), 2ones(n), -ones(n-1))) for _ in 1:d ]
        M   = KroneckerMatrix{SymInstance}(Mᵢ)
        b   = [ rand(n) for _ in 1:d ]
        nmax = 10

        #tensor_krylov_exact(M, b, 1e-9, nmax, TensorLanczos{Float64, SymInstance})

    end

end


@testset "Residual computations" begin

    d    = 3
    n    = 2
    rank = 2

    λ  = ones(rank)
    Y1 = [2.0 1.0; 1.0 2.0]
    Y2 = [3.0 4.0; 3.0 4.0]
    Y3 = [2.0 2.0; 2.0 2.0]
    A  = [Y1, Y2, Y3]
    y  = KruskalTensor{Float64}(A)

    v1 = Float64.([22, 22, 22, 22])
    v2 = Float64.([20, 20, 22, 22])
    v3 = Float64.([20, 20, 22, 22])

    manual_norms   = LinearAlgebra.norm.([v1, v2, v3]).^2
    computed_norms = zeros(d)

    Ly = [ zeros(rank, rank) for _ in 1:d ]
    Λ  = LowerTriangular( zeros(rank, rank) )

    compute_lower_triangles!(Ly, y)
    compute_lower_outer!(Λ, y.lambda)

    mask = trues(d)

    for s in 1:d

        Γ                 = cp_tensor_coefficients(Λ, y.fmat[s][n, :])
        mask[s]           = false
        computed_norms[s] = squared_tensor_entries(Ly[mask], Γ)
        mask[s]           = true

    end

    @test all(manual_norms .≈ computed_norms)

end

