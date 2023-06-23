using TensorKrylov: KroneckerMatrix, compute_lower_triangle!, innerproducts!, compressed_residual, matrix_vector, skipindex, efficient_matrix_vector_norm
using Kronecker, TensorToolbox, LinearAlgebra, TensorKrylov

# Last try of squared norm
function lastnorm(A::KroneckerMatrix{T}, x::ktensor) where T<:AbstractFloat

    d      = ndims(x)
    rank   = ncomponents(x)

    # Return vector of matrices as described above
    Z = matrix_vector(A, x)

    X_inner = [ ones(rank, rank) for _ in 1:d ]
    Z_inner = [ ones(rank, rank) for _ in 1:d ]
    ZX      = [ ones(rank, rank) for _ in 1:d ]
    

    for s = 1:d

        LinearAlgebra.mul!(X_inner[s], transpose(x.fmat[s]), x.fmat[s])
        LinearAlgebra.mul!(Z_inner[s], transpose(Z[s]), Z[s])
        LinearAlgebra.mul!(ZX[s], transpose(Z[s]), x.fmat[s])

    end

    my_norm = 0.0

    mask_s = trues(d)
    mask_r = trues(d)

    for s in 1:d

        for i = 1:rank

            my_norm += x.lambda[i]^2 * Z_inner[s][i, i]

            mask_s[s] = false

            for j = skipindex(i, 1:rank)

                my_norm += x.lambda[i] * x.lambda[j] * prod(getindex.(X_inner[mask_s], i, j)) * Z_inner[.!mask_s][1][i, j]

            end

        end

        for r = skipindex(s, 1:d)

            mask_r[r] = false

            for i = 1:rank

                my_norm += x.lambda[i]^2 * ZX[s][i, i] * ZX[r][i, i]

                for j = skipindex(i, 1:rank)

                    my_norm += x.lambda[i] * x.lambda[j] * prod(getindex.(ZX[mask_r .&& mask_s], i, j)) * prod(getindex.(ZX[.!(mask_r .&& mask_s)], j, i))

                end

            end

            mask_r[r] = true
        end

        mask_s[s] = true
    end

    return my_norm

end

# Everything here works
@testset "Lower triangle computations" begin

    d = 4

    t = 5

    #A = repeat( [rand(t, t)], d )
    A = rand(t, t)

    λ = rand(t)

    Λ = LowerTriangular(λ * λ')

    M = LowerTriangular(zeros(t, t))

    compute_lower_triangle!(M, λ)

    @test M ≈ Λ atol = 1e-15




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

    Λ = y.lambda * y.lambda'

    efficient_norm = efficient_matrix_vector_norm(y, Λ, Y_inner, B)

    lnorm = lastnorm(H, y)

    exact_matrix_vector = transpose( (H_kronsum * Y_vec) ) * (H_kronsum * Y_vec)

    @test efficient_norm ≈ lnorm


    @info "Exact norm:" exact_matrix_vector

    @info "Norm difference:" norm(full(y) - Y)

    @info "Norm difference between Y and its vectorization:" norm(Y) norm(Y_vec)


    #@test efficient_norm ≈ exact_matrix_vector

    
    # On the order of the machine precision
    exact_norm = norm(H_kronsum * Y_vec - b)

    # Check that we have indeed constructed a "good" low-rank approximation
    @test norm(full(y) - Y) < 1e-13

    # Create (lower triangular) matrices representing inner products
    LowerTriangles = repeat( [LowerTriangular( ones(rank, rank) )], d )

    innerproducts!(LowerTriangles, y.fmat, 1)

    computed_norm = compressed_residual(LowerTriangles, H, y, b)

end

