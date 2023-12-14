using TensorKrylov, Test

import TensorKrylov: compute_lower_outer!, compute_lower_triangles!, 
                     cp_tensor_coefficients, maskprod, matrix_vector, MVnorm, 
                     efficientMVnorm, compressed_residual, residual_norm,
                     squared_tensor_entries, tensorinnerprod

using Kronecker, TensorToolbox, LinearAlgebra, BenchmarkTools, SparseArrays

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

@testset "Compressed residual computations" begin

    function initialize_matrix_products(M, x)

        d      = length(M)
        t      = length(x.lambda)
        Λ      = LowerTriangular( zeros(t, t) )
        lowerX = [ zeros(t, t) for _ in 1:d ]
        Z      = matrix_vector(M, x)
        
        compute_lower_outer!(Λ, x.lambda)
        compute_lower_triangles!(lowerX, x)

        return Λ, lowerX, Z

    end


    function tensorsquarednorm(x::ktensor)

        d = length(x.fmat)
        t = length(x.lambda)

        value = 0.0 

        lowerX = [ zeros(t, t) for _ in 1:d ]
        
        compute_lower_triangles!(lowerX, x)

        X = Symmetric.(lowerX, :L)

        for j in 1:t, i in 1:t

            value += maskprod(X, i, j)

        end

        return value

    end

    @testset "Residual computation of linearly dependent factor matrices" begin

        d    = 3
        n    = 15
        rank = 4

        # Example of linearly dependent factor matrices in CP-decomposition
        Mᵢ  = [ Tridiagonal(-ones(n-1), 2ones(n), -ones(n-1)) for _ in 1:d ]
        M   = KroneckerMatrix{Float64}(Mᵢ)
        mat = rand(n, rank)
        X   = [ rand() .* mat for _ in 1:d ]
        #X   = [ rand(n, rank) for _ in 1:d ]
        x   = ktensor(ones(rank), X)

        Λ, lowerX, Z = initialize_matrix_products(M, x)

        for s in 1:d

            @test Mᵢ[s] * X[s] ≈ Z[s]

        end

        # Compute ||Mx||²
        exact_efficient_MVnorm = MVnorm(x, Λ, lowerX, Z)

        # Compute exact solution
        M_kroneckersum         = explicit_kroneckersum(Mᵢ)
        x_explicit             = kroneckervectorize(x)
        solution               = M_kroneckersum * x_explicit
        exactMVnorm            = dot(solution, solution)

        @test (exact_efficient_MVnorm - exactMVnorm) / exactMVnorm < 1e-14

        # Generate right-hand side
        b               = [ rand(n) for _ in 1:d ]
        b_explicit      = kron(b...)

        # Compute <Mx, b>₂
        approxinnerprod = tensorinnerprod(Z, x, b)
        exactinnerprod  = dot(solution, b_explicit)

        # If factor matrices in CP-decomposition are close to being orthogonal the dot product computation is very ill-conditioned.
        @test abs(exactinnerprod - approxinnerprod) / exactinnerprod < 1e-14

        b_norm = kronproddot(b)

        # Explicit compressed residual norm
        exp_comp_res_norm    = norm(b_explicit - solution)^2
        approx_comp_res_norm = compressed_residual(lowerX, Λ, M, x, b)
        relative_error       = (exp_comp_res_norm - approx_comp_res_norm) * inv(exp_comp_res_norm)
        @test relative_error  < 1e-15

    end

    @testset "Residual computation simulating method scenario" begin

        d = 3
        n = 15
        h = inv(n + 1)

        # Example of linearly dependent factor matrices in CP-decomposition
        Mᵢ  = [ sparse(inv(h^2) .* Tridiagonal(-ones(n-1), 2ones(n), -ones(n-1))) for _ in 1:d ]
        M   = KroneckerMatrix{Float64}(Mᵢ)
        b   = [ rand(n) for _ in 1:d ]
        nmax = 10

        #tensor_krylov_exact(M, b, 1e-9, nmax, TensorLanczos{Float64})

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
    y  = ktensor(A)

    v1 = Float64.([22, 22, 22, 22])
    v2 = Float64.([20, 20, 22, 22])
    v3 = Float64.([20, 20, 22, 22])

    manual_norms   = LinearAlgebra.norm.([v1, v2, v3]).^2
    computed_norms = zeros(d)

    Ly = [ zeros(rank, rank) for _ in 1:d ]
    Λ  = LowerTriangular( zeros(rank, rank) )

    compute_lower_triangles!(Ly, y)
    compute_lower_outer!(Λ, y.lambda)

    Ly   = Symmetric.(Ly, :L)
    mask = trues(d)

    for s in 1:d

        Γ                 = Symmetric(cp_tensor_coefficients(Λ, y.fmat[s][n, :]), :L)
        mask[s]           = false
        computed_norms[s] = squared_tensor_entries(Ly[mask], Γ)
        mask[s]           = true

    end

    @test all(manual_norms .≈ computed_norms)

end

