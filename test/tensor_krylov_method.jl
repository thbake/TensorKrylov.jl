using TensorKrylov: compute_lower_outer!, maskprod, compressed_residual, residual_norm, TensorDecomposition, kth_columns
using Kronecker, TensorToolbox, LinearAlgebra, BenchmarkTools, SparseArrays, ProfileView


# Everything here works
#@testset "Lower triangle computations" begin
#
#    d = 4
#
#    t = 5
#
#    A = rand(t, t)
#
#    λ = rand(t)
#
#    Λ = LowerTriangular(λ * λ')
#
#    M = LowerTriangular(zeros(t, t))
#
#    compute_lower_outer!(M, λ)
#
#    @test M ≈ Λ atol = 1e-15
#end
#
#@testset "Masked products" begin
#
#    n = 5
#    d = 4
#
#    matrices = [ rand(n, n) for _ in 1:d ]
#
#    values      = ones(n, n)
#    test_values = ones(n, n)
#
#    for j = 1:n, i = 1:n
#        
#        values[i, j] = maskprod(matrices, i, j) 
#
#    end
#
#    for s = 1:d
#
#        for j = 1:n, i = 1:n
#
#            test_values[i, j] *= matrices[s][i, j]
#
#        end
#
#    end
#
#    @test all(values .== test_values)
#
#    vec_matrix = [ rand(n, n) ]
#
#    for j = 1:n, i = 1:n
#
#        values[i, j] = maskprod(vec_matrix, i, j)
#
#    end
#
#    test_values .= copy(vec_matrix...)
#
#    @test all(values .== test_values)
#
#end
#
#@testset "(Compressed) residual norm computations" begin
#    
#    # We consider tensors of order 4, where each mode is 4 as well.
#    d = 5
#    nₛ= 5
#
#    Hᵢ= rand(nₛ, nₛ)
#
#    # Make sure matrix is not singular
#    H = KroneckerMatrix{Float64}([Hᵢ'Hᵢ, Hᵢ'Hᵢ, Hᵢ'Hᵢ, Hᵢ'Hᵢ, Hᵢ'Hᵢ])
#
#    # Matrix given as Kronecker sum
#    H_kronsum = kroneckersum( H.𝖳... )
#    
#    u = rand(nₛ)
#    v = rand(nₛ)
#    w = rand(nₛ)
#    x = rand(nₛ)
#    z = rand(nₛ)
#
#    # In the following we construct b as a rank 1 tensor such that the solution
#    # of the linear system H * y = b has a good low rank approximation.
#    b = zeros(nₛ, nₛ, nₛ, nₛ, nₛ)
#
#    for m = 1:nₛ, l = 1:nₛ, k = 1:nₛ, j = 1:nₛ, i = 1:nₛ
#
#        b[i, j, k, l, m] = u[i] * v[j] * w[k] * x[l] * z[m]
#
#    end
#
#    N = nₛ^d  
#
#    rank = 3
#
#    # Create Kruskal tensor such that there is no difference between this and its
#    # full tensor representation
#    y = ktensor( ones(rank), [ rand(d, rank) for _ in 1:d] )
#
#    Y_vec = reshape(full(y), N)
#
#    normalize!(y)
#
#    @info "Norm difference:" norm(reshape(full(y), N) - Y_vec)
#
#    # First test ||Hy||²
#    # Allocate memory for (lower triangular) matrices representing inner products
#    Y_inner = [ zeros(rank, rank) for _ in 1:d ]
#
#    for s = 1:d
#
#        LinearAlgebra.BLAS.syrk!('L', 'T', 1.0, y.fmat[s], 1.0, Y_inner[s])
#
#    end
#
#    Z = matrix_vector(H, y)
#
#    Λ = y.lambda * y.lambda'
#
#    Ly = [LowerTriangular( zeros(rank, rank) ) for _ in 1:d]
#
#    map!(LowerTriangular, Ly, Y_inner)
#
#    # Compute squared norm of Kronecker matrix and ktensor ||Hy||²
#    efficient_norm      = efficient_matrix_vector_norm(y, Λ, Ly, Z)
#    exact_matrix_vector = dot( (H_kronsum * Y_vec),  (H_kronsum * Y_vec) )
#
#    bY = [ zeros(1, rank) for _ in 1:d ] # bₛᵀyᵢ⁽ˢ⁾
#    bZ = [ zeros(1, rank) for _ in 1:d ] # bₛᵀzᵢ⁽ˢ⁾, where zᵢ⁽ˢ⁾ = Hₛ⋅yᵢ⁽ˢ⁾
#
#    # Right-hand side represented as factors of Kronecker product
#    b_kronprod = [u, v, w, x, z]
#
#    # Vectorization of right-hand side
#    b_vec = reshape(b, N)
#
#    # Compute inner product of Kronecker matrix times ktensor and right-hand side <Hy, b>
#    innerprod        = innerprod_kronsum_tensor!(bY, bZ, Z, y, b_kronprod)
#    exact_innerprod  = dot(H_kronsum * Y_vec, b_vec)
#
#    @test efficient_norm ≈ exact_matrix_vector 
#    @test innerprod      ≈ dot(H_kronsum * Y_vec, b_vec) atol = 1e-12 
#
#    # Compressed residual norm
#    r_comp = compressed_residual(Ly, LowerTriangular(Λ), H, y, b_kronprod)
#
#    exact_comp_norm = exact_matrix_vector - 2 * dot(H_kronsum * Y_vec, b_vec) + dot(b_vec, b_vec)
#    
#    @info exact_comp_norm
#    @test r_comp ≈ exact_comp_norm 
#
#
#    res_norm = residual_norm(H, y, [3, 3, 3, 3, 3], b_kronprod)
#
#    @info abs(res_norm - exact_comp_norm)
#
#    @info cond(H_kronsum)
#
#
#    #@info "Exact ||Hy||²: " exact_matrix_vector " exact 2 ⋅<Hy, b>: " 2*dot(H_kronsum*Y_vec, b_vec) " exact ||b||²: " dot(b_vec, b_vec)
#    
#    # On the order of the machine precision
#
#    # Check that we have indeed constructed a "good" low-rank approximation
#    #@test norm(full(y) - Y) < 1e-13
#
#end

@testset "Residual computations" begin

    function solvecompressed(H::KroneckerMatrix{T}, b::Vector{<:AbstractVector{T}}) where T<:AbstractFloat
    #function solvecompressed(H::KroneckerMatrix{T}, b::Vector{T}) where T<:AbstractFloat

        H_expanded = Symmetric(kroneckersum(H.𝖳...))
        #H_expanded = Symmetric(Matrix(H.𝖳...))
        b_expanded = kron(b...)
        y          = H_expanded\b_expanded
        #y          = H_expanded\b

        return y

    end

    function exactresidualnorm(A::KroneckerMatrix{T}, b::Vector{<:AbstractVector{T}}, xₖ::AbstractVector{T}) where T<:AbstractFloat
        A_expanded = kroneckersum(A.𝖳...)
        #A_expanded = A.𝖳[1]
        b_expanded = kron(b...)
        #b_expanded = b[1]
        tmp        = zeros(size(A_expanded, 1))

        mul!(tmp, A_expanded, xₖ)

        rₖ = b_expanded - tmp

        return sqrt(dot(rₖ, rₖ)) * inv(LinearAlgebra.norm(b_expanded))

    end

    function Anormerror(A::AbstractMatrix{T}, x::AbstractVector{T}, xₖ::AbstractVector{T}) where T<: AbstractFloat

        tmp = zeros(size(x)) 
        diff = x - xₖ

        mul!(tmp, A, diff)

        return sqrt(dot(diff, diff))

    end

    function tensor_krylov_exact(
            A::KroneckerMatrix{T},
            b::Vector{<:AbstractVector{T}},
            nmax::Int,
            t_orthonormalization::Type{<:TensorDecomposition}) where T <: AbstractFloat

        d = length(A)
        xₖ = Vector{T}(undef, nentries(A))
        b̃ = [ zeros( size(b[s]) )  for s in eachindex(b) ]

        A_expanded = kroneckersum(A.𝖳...)
        #A_expanded = A.𝖳[1]
        b_expanded = kron(b...)
        #b_expanded = b[1]

        x = Symmetric(A_expanded)\b_expanded

        tensor_decomp      = t_orthonormalization(A)
        orthonormalization = tensor_decomp.orthonormalization

        initial_orthonormalization!(tensor_decomp, b, orthonormalization)
        
        for k = 2:nmax

            orthonormal_basis!(tensor_decomp, k)

            H_minors = principal_minors(tensor_decomp.H, k)
            V_minors = principal_minors(tensor_decomp.V, n, k)
            b_minors = principal_minors(b̃, k)
            #b̃ = zeros(k)

            # Update compressed right-hand side b̃ = Vᵀb
            update_rhs!(b_minors, V_minors, b, k)
            #b̃ = transpose(V) * b_expanded

            y = solvecompressed(H_minors, b_minors)

            mul!(xₖ, kron(V_minors.𝖳...), y)
            #mul!(xₖ, V_minors.𝖳[1], y)

            r_normexact = exactresidualnorm(A, b, xₖ)

            println(r_normexact)

            error = Anormerror(A_expanded, x, xₖ)

            @info "Error x - xₖ" error

            #residual_norm(H_minors, y)

        end

    end

    d    = 2
    n    = 10
    h    = inv(n + 1)
    Tₖ   = inv(h^2) .* (Tridiagonal(-ones(n - 1), 2ones(n), -ones(n - 1)))
    A    = KroneckerMatrix{Float64}([Tₖ for _ in 1:d])
    b    = [ rand(n) for _ in 1:d ]
    nmax = 9

    tensor_krylov_exact(A, b, nmax, TensorLanczos{Float64})

end

#@testset "Analytic results" begin
#
#    d = 2
#
#    nₛ = 10
#
#
#    h = inv(nₛ + 1)
#
#    Aₛ= sparse(
#
#            Tridiagonal( -1ones(nₛ - 1) , 2ones(nₛ), -1ones(nₛ - 1) )
#        )
#
#    #Aₛ= sparse( Tridiagonal( -1ones(nₛ - 1) , 2ones(nₛ), -1ones(nₛ - 1) ) )
#
#    #A = KroneckerMatrix{Float64}([Aₛ for _ in 1:d])
#    A = trikronmat([nₛ for _ in 1:d])
#
#    b = [ rand(nₛ) for _ in 1:d ]
#
#    τ = 1e-14
#
#    #λ_min = (2 / h^2) * (1 - cos( π / (nₛ + 1)))
#    #λ_max = (2 / h^2) * (1 - cos( nₛ * π / (nₛ + 1)))
#
#    λ_min = d * 2(1 - cos( π / (nₛ + 1)))
#    λ_max = d * 2(1 - cos( nₛ * π / (nₛ + 1)))
#
#    A_big = kroneckersum(A.𝖳...)
#
#    julia_eigenvalues = eigvals(A_big)
#
#    @test λ_min ≈ julia_eigenvalues[1]
#    @test λ_max ≈ julia_eigenvalues[end]
#
#    #κ = 4 * (nₛ + 1)^2 / (π^2 * d)
#    #κ = 1 + cos(π / (nₛ + 1)) * inv( d * (1 - cos(π / (nₛ + 1)) ))
#
#    κ = λ_max / λ_min
#
#    @test κ ≈ cond(A_big)
#
#    @assert issparse(A_big)
#
#end

@testset "Symmetric example" begin

    d = 5
    nₛ = 200
    nmax = 190

    h = inv(nₛ + 1)

    Aₛ= sparse(inv(h^2) * Tridiagonal( -1ones(nₛ - 1) , 2ones(nₛ), -1ones(nₛ - 1) ))

    A = KroneckerMatrix{Float64}([Aₛ for _ in 1:d])

    b = [ rand(nₛ) for _ in 1:d ]
    
    for s in eachindex(b)

        b[s] = inv(LinearAlgebra.norm(b[s])) .* b[s]

    end

    b_norm = kronprodnorm(b)

    @info "Norm of ⨂ b " b_norm

    tensor_krylov(A, b, 1e-6, nmax, TensorLanczos{Float64})

end

#@testset "Nonsymmetric example" begin
#
#    d = 5
#    n = 50
#    nmax = 49
#    h = inv(n + 1)
#    c = 10
#
#    L  = sparse( inv(h^2) .* Tridiagonal(-ones(n - 1), 2ones(n), -ones(n - 1)) )
#    Aₛ = L + sparse( (c / (4 * h)) .* diagm(-1 => ones(n-1), 0 => 3ones(n), 1 => -5ones(n - 1), 2 => ones(n - 2)) )
#
#    A = KroneckerMatrix{Float64}([Aₛ for _ in 1:d])
#
#    b = [ rand(nₛ) for _ in 1:d ]
#    
#    for s in eachindex(b)
#
#        b[s] = inv(LinearAlgebra.norm(b[s])) .* b[s]
#
#    end
#    
#    x = tensor_krylov(A, b, 1e-9, nmax, TensorArnoldi{Float64})
#
#
#    
