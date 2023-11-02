using TensorKrylov, Test
using Kronecker, TensorToolbox, LinearAlgebra, BenchmarkTools, SparseArrays, ProfileView

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

@testset "Tensor computations" begin

    d    = 3
    n    = 15
    rank = 4

    Mᵢ= [ Tridiagonal(-ones(n-1), 2ones(n), -ones(n-1)) for _ in 1:d ]
    M = KroneckerMatrix{Float64}(Mᵢ)
    X = [ rand(n, rank) for _ in 1:d ]
    x = ktensor(ones(rank), X)

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

    function compute_exactMVnorm(x, Λ, lowerX, Z)
        
        Λ_complete = Symmetric(Λ, :L)
        X          = Symmetric.(lowerX, :L)
        Z_inner    = [ Z[s]'Z[s] for s in 1:length(Z) ]
        XZ         = [ x.fmat[s]'Z[s] for s in 1:ndims(x) ]

        d    = length(lowerX)
        rank = length(x.lambda)

        mask_s = falses(d)
        mask_r = falses(d)

        MVnorm = 0.0

        for j = 1:rank, i = 1:rank

            for s in 1:d

                mask_s[s] = true

                for r in 1:d

                    mask_r[r] = true

                    MVnorm += Λ_complete[i, j] * maskprod( X[.!(mask_s .|| mask_r)], i, j ) *  maskprod(XZ[mask_s .⊻ mask_r], i, j) * maskprod(Z_inner[mask_s .&& mask_r], i, j)

                    mask_r[r] = false
                end

                mask_s[s] = false

            end

        end

        return MVnorm
        
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


    Λ, lowerX, Z = initialize_matrix_products(M, x)

    for s in 1:d

        @test Mᵢ[s] * X[s] ≈ Z[s]

    end

    MVnorm      = efficientMVnorm(x, Λ, lowerX, Z)
    exact_efficient_MVnorm = compute_exactMVnorm(x, Λ, lowerX, Z)

    M_kroneckersum = explicit_kroneckersum(Mᵢ)
    #x_outer     = reshape(full(x), n^d)
    x_explicit     = kroneckervectorize(x)
    
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

    tnorm       = tensorsquarednorm(x)
    ex_norm     = dot(x_explicit, x_explicit)
    solution    = M_kroneckersum * x_explicit
    exactMVnorm = dot(solution, solution)

    @info (tnorm - ex_norm) / ex_norm
    @test (exact_efficient_MVnorm - exactMVnorm) / exactMVnorm < 1e-15





end

#@testset "(Compressed) residual norm computations" begin
#    
#    # We consider tensors of order 4, where each mode is 4 as well.
#    d = 5
#    nₛ= 5
#
#    Hᵢ        = sparse( Tridiagonal(-ones(nₛ - 1), 2ones(nₛ), -ones(nₛ - 1)) )
#    H         = KroneckerMatrix{Float64}([Hᵢ for _ in 1:d])
#    H_kronsum = explicit_kroneckersum( [Hᵢ for _ in 1:d ])
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
#    @assert norm(reshape(full(y), N) - Y_vec) / norm(Y_vec) < 1e-15
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
#    efficient_norm      = efficientMVnorm(y, Λ, Ly, Z)
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
#    @info norm(r_comp - exact_comp_norm) / norm(exact_comp_norm)
#    @test r_comp ≈ exact_comp_norm 
#
#    𝔎 = [ 3 for _ in 1:d ]
#
#    subdiagonal_entries = [ H[s][𝔎[s] + 1, 𝔎[s]] for s in 1:d ]
#
#    res_norm = residual_norm(H, y, 𝔎, subdiagonal_entries, b_kronprod)
#
#    #@info "Differene between abs(res_norm - exact_comp_norm)
#    Y = H_kronsum\( reshape(b, N) )
#
#    #@info "Exact ||Hy||²: " exact_matrix_vector " exact 2 ⋅<Hy, b>: " 2*dot(H_kronsum*Y_vec, b_vec) " exact ||b||²: " dot(b_vec, b_vec)
#    
#    # On the order of the machine precision
#
#    # Check that we have indeed constructed a "good" low-rank approximation
#    #@test norm( reshape(full(y), N) - Y) < 1e-13
#    @info norm( reshape(full(y), N) - Y)  / norm(Y)
#
#end


#@testset "Residual computations" begin
#
#    d    = 3
#    n    = 2
#    rank = 2
#
#    λ  = ones(rank)
#    Y1 = [2.0 1.0; 1.0 2.0]
#    Y2 = [3.0 4.0; 3.0 4.0]
#    Y3 = [2.0 2.0; 2.0 2.0]
#    A  = [Y1, Y2, Y3]
#    y  = ktensor(A)
#
#    v1 = Float64.([22, 22, 22, 22])
#    v2 = Float64.([20, 20, 22, 22])
#    v3 = Float64.([20, 20, 22, 22])
#
#    manual_norms   = LinearAlgebra.norm.([v1, v2, v3]).^2
#    computed_norms = zeros(d)
#
#    Ly = [ zeros(rank, rank) for _ in 1:d ]
#
#    compute_lower_triangles!(Ly, y)
#
#    Λ = LowerTriangular( zeros(rank, rank) )
#
#    compute_lower_outer!(Λ, y.lambda)
#
#    Ly = Symmetric.(Ly, :L)
#
#    mask = trues(d)
#
#    for s in 1:d
#
#        Γ                 = Symmetric(compute_coefficients(Λ, y.fmat[s][n, :]), :L)
#        mask[s]           = false
#        computed_norms[s] = squared_tensor_entries(Ly[mask], Γ)
#        mask[s]           = true
#
#    end
#
#    @test all(manual_norms .≈ computed_norms)
#
#end
#
#@testset "Monotonic decrease of residual and error in A-norm" begin
#
#    function solvecompressed(H::KroneckerMatrix{T}, b::Vector{<:AbstractVector{T}}) where T<:AbstractFloat
#
#        H_expanded = Symmetric(kroneckersum(H.𝖳...))
#        b_expanded = kron(b...)
#        y          = H_expanded\b_expanded
#
#        return y
#
#    end
#
#    function exactresidualnorm(A::KroneckerMatrix{T}, b::Vector{<:AbstractVector{T}}, xₖ::AbstractVector{T}) where T<:AbstractFloat
#        A_expanded = kroneckersum(A.𝖳...)
#        b_expanded = kron(b...)
#        tmp        = zeros(size(A_expanded, 1))
#
#        mul!(tmp, A_expanded, xₖ)
#
#        rₖ = b_expanded - tmp
#
#        return sqrt(dot(rₖ, rₖ)) * inv(LinearAlgebra.norm(b_expanded))
#
#    end
#
#    function Anormerror(A::AbstractMatrix{T}, x::AbstractVector{T}, xₖ::AbstractVector{T}) where T<: AbstractFloat
#
#        tmp = zeros(size(x)) 
#        diff = x - xₖ
#
#        mul!(tmp, A, diff)
#
#        return sqrt(dot(diff, diff))
#
#    end
#
#    function tensor_krylov_exact(
#            A::KroneckerMatrix{T},
#            b::Vector{<:AbstractVector{T}},
#            nmax::Int,
#            t_orthonormalization::Type{<:TensorDecomposition}) where T <: AbstractFloat
#
#        xₖ = Vector{T}(undef, nentries(A))
#
#        A_expanded = kroneckersum(A.𝖳...)
#        b_expanded = kron(b...)
#
#        x = Symmetric(A_expanded)\b_expanded
#
#        tensor_decomp      = t_orthonormalization(A)
#        orthonormalization = tensor_decomp.orthonormalization
#
#        initial_orthonormalization!(tensor_decomp, b, orthonormalization)
#
#        n = size(A[1], 1)
#        println(n)
#
#        b̃  = initialize_compressed_rhs(b, tensor_decomp.V)
#        
#        for k = 2:nmax
#
#            orthonormal_basis!(tensor_decomp, k)
#
#            H_minors = principal_minors(tensor_decomp.H, k)
#            V_minors = principal_minors(tensor_decomp.V, n, k)
#            b_minors = principal_minors(b̃, k)
#
#            # Update compressed right-hand side b̃ = Vᵀb
#            columns = kth_columns(tensor_decomp.V, k)
#
#            update_rhs!(b_minors, columns, b, k)
#
#            y  = solvecompressed(H_minors, b_minors)
#
#            mul!(xₖ, kron(V_minors.𝖳...), y)
#
#            r_normexact = exactresidualnorm(A, b, xₖ)
#
#            println(r_normexact)
#
#            error = Anormerror(A_expanded, x, xₖ)
#
#            @info "Error x - xₖ" error
#
#            #residual_norm(H_minors, y)
#
#        end
#
#    end
#
#    d    = 2
#    n    = 100
#    h    = inv(n + 1)
#    Tₖ   = sparse(inv(h^2) .* (Tridiagonal(-ones(n - 1), 2ones(n), -ones(n - 1))))
#    A    = KroneckerMatrix{Float64}([Tₖ for _ in 1:d])
#    b    = [ rand(n) for _ in 1:d ]
#    nmax = 190
#
#    tensor_krylov_exact(A, b, nmax, TensorLanczos{Float64})
#
#end

#@testset "Symmetric example" begin
#
#    d = 50
#    nₛ = 200
#    nmax = 100
#
#    h = inv(nₛ + 1)
#
#    Aₛ= sparse(inv(h^2) * Tridiagonal( -1ones(nₛ - 1) , 2ones(nₛ), -1ones(nₛ - 1) ))
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
#    b_norm = kronprodnorm(b)
#
#    @info "Norm of ⨂ b " b_norm
#
#    tensor_krylov(A, b, 1e-6, nmax, TensorLanczos{Float64})
#
#end

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
