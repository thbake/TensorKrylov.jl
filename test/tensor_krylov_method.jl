using TensorKrylov: KroneckerMatrix, compute_lower_triangle!, innerproducts!, compressed_residual, matrix_vector, lastnorm, efficient_matrix_vector_norm
using Kronecker, TensorToolbox, LinearAlgebra, TensorKrylov


# Everything here works
#@testset "Lower triangle computations" begin
#
#    d = 4
#
#    t = 5
#
#    A = repeat( [rand(t, t)], d )
#
#    LowerTriangles = [ LowerTriangular(ones(t, t)) for _ in 1:d ]
#
#    L₀     = LowerTriangles[1]
#    L₁     = LowerTriangles[2]
#
#    compute_lower_triangle!(L₀, A[1], 0) # Including diagonal
#    compute_lower_triangle!(L₁, A[2], 1) # Below the diagonal
#
#
#    exact₀ = LowerTriangular(A[1]'A[1])
#    exact₁ = LowerTriangular(A[2]'A[2])
#
#    exact₁[diagind(exact₁)] = ones(t)
#
#    @test L₀ ≈ exact₀ atol = 1e-12
#    @test L₁ ≈ exact₁ atol = 1e-12
#
#    innerproducts!(LowerTriangles, A, 0)
#
#    for s = 1:d
#
#        @test LowerTriangles[s] ≈ LowerTriangular(A[s]'A[s]) atol = 1e-12 
#
#    end
#
#end

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

    # Solve the system directly
    Y_vec = H_kronsum \ reshape(b, N)

    # Express solution as d-way tensor
    Y = reshape(Y_vec, (nₛ, nₛ, nₛ, nₛ))

    # Rank 3 Kruskal tensor (CP) with vectors of order 4.
    rank = 3

    # Construct low- (three-) rank decomposition of Y.
    y = cp_als(Y, rank) 
    
    # First test ||Hy||²
    # Allocate memory for (lower triangular) matrices representing inner products
    Y_inner = [ zeros(rank, rank) for _ in 1:d ]

    for s = 1:d

        LinearAlgebra.BLAS.syrk!('L', 'T', 1.0, y.fmat[s], 1.0, Y_inner[s])

    end

    B = matrix_vector(H, y)

    efficient_norm = efficient_matrix_vector_norm(H, y, Y_inner, B)

    lnorm = lastnorm(H, y)

    @info "Last norm:" lnorm 

    @info "Efficient norm:" efficient_norm

    @info "Exact norm:" transpose((H_kronsum * Y_vec)) * (H_kronsum * Y_vec)

    @info "Norm difference:" norm(full(y) - Y)

    @info "Norm difference between Y and its vectorization:" norm(Y) norm(Y_vec)


    @test squared_norm ≈ exact_matrix_vector

    
    # On the order of the machine precision
    exact_norm = norm(H_kronsum * Y_vec - b)

    # Check that we have indeed constructed a "good" low-rank approximation
    @test norm(full(y) - Y) < 1e-13

    # Create (lower triangular) matrices representing inner products
    LowerTriangles = repeat( [LowerTriangular( ones(rank, rank) )], d )

    innerproducts!(LowerTriangles, y.fmat, 1)

    computed_norm = compressed_residual(LowerTriangles, H, y, b)

end

