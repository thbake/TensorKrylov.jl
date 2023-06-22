using TensorKrylov: KroneckerMatrix, compute_lower_triangle!, innerproducts!, ktensor_innerprods!, compressed_residual, matrix_vector, squared_matrix_vector, recursivekronecker, squared_norm_vectorized, lastnorm
using Kronecker, TensorToolbox, LinearAlgebra, TensorKrylov


#function recursive_norm(
#        X,
#        Z,
#        XZ,
#        s, r, i, j d::Int)
#    j
#    if d == 1
#
#        return y
#
#    elseif s == 1 && d > 1
#
#        return recursive_norm(x, y, s, d - 1) * x
#
#    else
#
#        return x * recursive_norm(x, y, s - 1, d - 1)
#
#    end
#
#end

function exact_squared_norm(x::ktensor, Z::Vector{Matrix{T}}, d::Int, N::Int) where T <: AbstractFloat
    
    rank = ncomponents(x)

    lhs = zeros(N)

    for s = 1:d, i = 1:rank

        lhs += recursivekronecker(Z, x.lambda[i] * x.fmat, s, i, d)

    end

    rhs = zeros(N)

    for r = 1:d, j = 1:rank

        rhs += recursivekronecker(Z, x.lambda[j] * x.fmat, r, j, d)

    end

    return lhs'rhs
end

function exact_squared_different(X, Z, XZ, d::Int)

    rank = size(X, 1)

    #for s = 1:d, r = 1:d, j = 1:rank, i = 1:rank

    #X[s][i, j] XZ[s][i, j]

    #    recursive_norm(X[s][i, j], Z[s][i, j])

    #end

        
end

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

    # For this we pre-compute a (lower triangular) matrix that represents inner
    # products of the form yᵢ⁽ˢ⁾ᵀyⱼ⁽ˢ⁾
    Ly = repeat( [LowerTriangular( ones(rank, rank) )], d )

    ktensor_innerprods!(Ly, y)

    B = matrix_vector(H, y)

    squared_norm = squared_matrix_vector(Ly, B, H, y)

    exact_matrix_vector = 0.0

    exact_matrix_vector = exact_squared_norm(y, B, d, N)

    vectorized_norm = squared_norm_vectorized(y, H)

    lnorm = lastnorm(H, y)

    @info "Efficient norm:" squared_norm

    @info "Recursive Kronecker norm:" exact_matrix_vector

    @info "Vectorized norm:" vectorized_norm

    @info "Last norm:" lnorm 

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

