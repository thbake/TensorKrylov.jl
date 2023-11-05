using TensorKrylov, Test
using Kronecker, TensorToolbox, LinearAlgebra, BenchmarkTools, SparseArrays, ProfileView

@testset "Monotonic decrease of residual and error in A-norm" begin

    function solvecompressed(H::KroneckerMatrix{T}, b::Vector{<:AbstractVector{T}}) where T<:AbstractFloat

        H_expanded = Symmetric(kroneckersum(H.𝖳...))
        b_expanded = kron(b...)
        y          = H_expanded\b_expanded

        return y

    end

    function exactresidualnorm(A::KroneckerMatrix{T}, b::Vector{<:AbstractVector{T}}, xₖ::AbstractVector{T}) where T<:AbstractFloat
        A_expanded = kroneckersum(A.𝖳...)
        b_expanded = kron(b...)
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

    function tensor_krylov_exact(A::KronMat{T}, b::KronProd{T}, nmax::Int, t_orthonormalization::Type{<:TensorDecomposition}) where T <: AbstractFloat

        xₖ = Vector{T}(undef, nentries(A))

        A_expanded = kroneckersum(A.𝖳...)
        b_expanded = kron(b...)

        x = Symmetric(A_expanded)\b_expanded

        tensor_decomp      = t_orthonormalization(A)
        orthonormalization = tensor_decomp.orthonormalization

        initial_orthonormalization!(tensor_decomp, b, orthonormalization)

        n = size(A[1], 1)
        println(n)

        b̃  = initialize_compressed_rhs(b, tensor_decomp.V)
        
        for k = 2:nmax

            orthonormal_basis!(tensor_decomp, k)

            H_minors = principal_minors(tensor_decomp.H, k)
            V_minors = principal_minors(tensor_decomp.V, n, k)
            b_minors = principal_minors(b̃, k)

            # Update compressed right-hand side b̃ = Vᵀb
            columns = kth_columns(tensor_decomp.V, k)

            update_rhs!(b_minors, columns, b, k)

            y  = solvecompressed(H_minors, b_minors)

            mul!(xₖ, kron(V_minors.𝖳...), y)

            r_normexact = exactresidualnorm(A, b, xₖ)

            println(r_normexact)

            error = Anormerror(A_expanded, x, xₖ)

            @info "Error x - xₖ" error

            #residual_norm(H_minors, y)

        end

    end

    d    = 2
    n    = 100
    h    = inv(n + 1)
    Tₖ   = sparse(inv(h^2) .* (Tridiagonal(-ones(n - 1), 2ones(n), -ones(n - 1))))
    A    = KroneckerMatrix{Float64}([Tₖ for _ in 1:d])
    b    = [ rand(n) for _ in 1:d ]
    nmax = 190

    #tensor_krylov_exact(A, b, nmax, TensorLanczos{Float64})

end

@testset "Symmetric example" begin

    d = 50
    nₛ = 200
    nmax = 100

    h = inv(nₛ + 1)

    Aₛ= sparse(inv(h^2) * Tridiagonal( -1ones(nₛ - 1) , 2ones(nₛ), -1ones(nₛ - 1) ))

    A = KroneckerMatrix{Float64}([Aₛ for _ in 1:d])

    b = [ rand(nₛ) for _ in 1:d ]
    
    #for s in eachindex(b)

    #    b[s] = inv(LinearAlgebra.norm(b[s])) .* b[s]

    #end

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
