using TensorKrylov, Test

import TensorKrylov: compute_lower_outer!, compute_lower_triangles!, 
                     compute_coefficients, maskprod, matrix_vector, MVnorm, 
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

    d    = 3
    n    = 15
    rank = 4

    Mᵢ  = [ Tridiagonal(-ones(n-1), 2ones(n), -ones(n-1)) for _ in 1:d ]
    M   = KroneckerMatrix{Float64}(Mᵢ)
    mat = rand(n, rank)
    X   = [ rand() .* mat for _ in 1:d ]
    x   = ktensor(ones(rank), X)

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

    function kroneckervectorize(x::ktensor)

        N    = prod(size(x))
        vecx = zeros(N)

        for i in 1:ncomponents(x) 

            tmp = @view(x.fmat[end][:, i])

            for j in ndims(x) - 1 : - 1 : 1

                tmp = kron(tmp, @view(x.fmat[j][:, i]))

            end

            vecx += tmp

        end

        return vecx

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

    @test (exp_comp_res_norm  - approx_comp_res_norm) / exp_comp_res_norm < 1e-14

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

    compute_lower_triangles!(Ly, y)

    Λ = LowerTriangular( zeros(rank, rank) )

    compute_lower_outer!(Λ, y.lambda)

    Ly = Symmetric.(Ly, :L)

    mask = trues(d)

    for s in 1:d

        Γ                 = Symmetric(compute_coefficients(Λ, y.fmat[s][n, :]), :L)
        mask[s]           = false
        computed_norms[s] = squared_tensor_entries(Ly[mask], Γ)
        mask[s]           = true

    end

    @test all(manual_norms .≈ computed_norms)

end

