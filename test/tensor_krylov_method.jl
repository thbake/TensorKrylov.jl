using TensorKrylov: KroneckerMatrix, compute_lower_triangles!, compute_lower_outer!, compressed_residual, matrix_vector, skipindex, efficient_matrix_vector_norm, residual_norm, maskprod, innerprod_kronsum_tensor!
using Kronecker, TensorToolbox, LinearAlgebra, TensorKrylov


# Everything here works
@testset "Lower triangle computations" begin

    d = 4

    t = 5

    A = rand(t, t)

    λ = rand(t)

    Λ = LowerTriangular(λ * λ')

    M = LowerTriangular(zeros(t, t))

    compute_lower_outer!(M, λ)

    @test M ≈ Λ atol = 1e-15
end

@testset "Masked products" begin

    n = 5
    d = 4

    matrices = [ rand(n, n) for _ in 1:d ]

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

@testset "Compressed residual norm" begin
    
    # We consider tensors of order 4, where each mode is 4 as well.
    d = 4
    nₛ= 4

    Hᵢ= rand(nₛ, nₛ)

    # Make sure matrix is not singular
    H = KroneckerMatrix([Hᵢ'Hᵢ, Hᵢ'Hᵢ, Hᵢ'Hᵢ, Hᵢ'Hᵢ])

    # Matrix given as Kronecker sum
    H_kronsum = kroneckersum( H.𝖳... )
    
    u = rand(nₛ)
    v = rand(nₛ)
    w = rand(nₛ)
    x = rand(nₛ)

    # In the following we construct b as a rank 1 tensor such that the solution
    # of the linear system H * y = b has a good low rank approximation.
    b = zeros(nₛ, nₛ, nₛ, nₛ)

    for l = 1:nₛ, k = 1:nₛ, j = 1:nₛ, i = 1:nₛ

        b[i, j, k, l] = u[i] * v[j] * w[k] * x[l]

    end

    N = nₛ^d  

    # ============================================================
    # Ignore this for a moment

    # Solve the system directly
    #Y_vec = H_kronsum \ reshape(b, N)

    # Express solution as d-way tensor
    #Y = reshape(Y_vec, (nₛ, nₛ, nₛ, nₛ))

    # Rank 3 Kruskal tensor (CP) with vectors of order 4.
    #rank = 3

    # Construct low- (three-) rank decomposition of Y.
    #y = cp_als(Y, rank) 
    
    # ===========================================================

    rank = 3

    y = ktensor( ones(rank), [ rand(d, rank) for _ in 1:d] )

    Y_vec = reshape(full(y), N)

    normalize!(y)

    @info "Norm difference:" norm(reshape(full(y), N) - Y_vec)

    # First test ||Hy||²
    # Allocate memory for (lower triangular) matrices representing inner products
    Y_inner = [ zeros(rank, rank) for _ in 1:d ]

    for s = 1:d

        LinearAlgebra.BLAS.syrk!('L', 'T', 1.0, y.fmat[s], 1.0, Y_inner[s])

    end

    Z = matrix_vector(H, y)

    Λ = y.lambda * y.lambda'


    Ly = [LowerTriangular( zeros(rank, rank) ) for _ in 1:d]
    map!(LowerTriangular, Ly, Y_inner)

    efficient_norm      = efficient_matrix_vector_norm(y, Λ, Ly, Z)
    exact_matrix_vector = dot( (H_kronsum * Y_vec),  (H_kronsum * Y_vec) )

    bY = [ zeros(1, rank) for _ in 1:d ] # bₛᵀyᵢ⁽ˢ⁾
    bZ = [ zeros(1, rank) for _ in 1:d ] # bₛᵀzᵢ⁽ˢ⁾, where zᵢ⁽ˢ⁾ = Hₛ⋅yᵢ⁽ˢ⁾

    # Right-hand side represented as factors of Kronecker product
    b_kronprod = [u, v, w, x]
    b_vec = reshape(b, N)

    innerprod        = innerprod_kronsum_tensor!(bY, bZ, Z, y, b_kronprod)
    exact_innerprod  = dot(H_kronsum * Y_vec, b_vec)

    @test efficient_norm ≈ exact_matrix_vector atol = 1e-11
    @test innerprod      ≈ dot(H_kronsum * Y_vec, b_vec) atol = 1e-12 

    # Compressed residual norm
    r_comp = compressed_residual(Ly, LowerTriangular(Λ), H, y, b_kronprod)

    exact_comp_norm = exact_matrix_vector - 2 * dot(H_kronsum * Y_vec, b_vec) + dot(b_vec, b_vec)
    
    @info exact_comp_norm
    @test r_comp ≈ exact_comp_norm atol = 1e-11


    #@info "Exact ||Hy||²: " exact_matrix_vector " exact 2 ⋅<Hy, b>: " 2*dot(H_kronsum*Y_vec, b_vec) " exact ||b||²: " dot(b_vec, b_vec)
    
    # On the order of the machine precision
    exact_norm = dot((H_kronsum * Y_vec - b_vec),  (H_kronsum * Y_vec - b_vec))
    @info exact_norm

    # Check that we have indeed constructed a "good" low-rank approximation
    #@test norm(full(y) - Y) < 1e-13

end

