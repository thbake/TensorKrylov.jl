export Arnoldi, Lanczos, TensorArnoldi, TensorLanczos, TensorDecomposition
export orthonormal_basis!, orthonormal_basis_vector!, initial_orthonormalization!, initialize_decomp!, isorthonormal

const MatrixView{T}    = AbstractMatrix{T}

abstract type Decomposition{T}       end
abstract type TensorDecomposition{T} end


function order(decomposition::Decomposition)

    return size(decomposition.A, 1)

end

function Base.length(tensor_decomp::TensorDecomposition) 

    return length(tensor_decomp.A)

end

mutable struct Arnoldi{T} <: Decomposition{T}

    A::MatrixView{T}
    V::MatrixView{T}
    H::MatrixView{T}

    function Arnoldi{T}(
                A::MatrixView{T},
                V::MatrixView{T},
                H::MatrixView{T},
                b::AbstractArray{T}) where T<:AbstractFloat

        initialize_decomp!(V, b)

        new(A, V, H)

    end

    function Arnoldi{T}(
                A::MatrixView{T},
                V::MatrixView{T},
                H::MatrixView{T}) where T<:AbstractFloat

        new(A, V, H)

    end
                       
end

mutable struct Lanczos{T} <: Decomposition{T}

    A::MatrixView{T}
    V::MatrixView{T}
    H::MatrixView{T}
    β::T                # Last off-diagonal coefficient in H
    v::AbstractArray{T} # Last basis vector

    function Lanczos{T}(
                A::MatrixView{T},
                V::MatrixView{T},
                H::MatrixView{T},
                b::AbstractArray{T}) where T<:AbstractFloat

        # β₀ = 0, v₀ = 0
        β = 0.0
        v = zeros(size(A, 1)) # vₖ₋₁

        initialize_decomp!(V, b)

        new(A, V, H, β, v)

    end

    function Lanczos{T}(
                A::MatrixView{T},
                V::MatrixView{T},
                H::MatrixView{T},
                k::Int) where T<:AbstractFloat

        β = H[k - 1, k]
        v = @view(V[:, k - 1])

        new(A, V, H, β, v)

    end

end

function initialize_decomp!(V::MatrixView{T}, b::AbstractArray{T}) where T<:AbstractFloat

        # Normalize first basis vector

        V[:, 1] = inv( LinearAlgebra.norm(b) ) .* b

end

mutable struct TensorArnoldi{T} <: TensorDecomposition{T}

    A::KroneckerMatrix{T} # Original matrix
    V::KroneckerMatrix{T} # Matrix representing basis of Krylov subspace
    H::KroneckerMatrix{T} # Upper Hessenberg matrix
    orthonormalization::Type{Arnoldi}

    function TensorArnoldi{T}(A::KroneckerMatrix{T}) where T<:AbstractFloat

        V = KroneckerMatrix{T}(dimensions(A))
        H = KroneckerMatrix{T}(

            [ UpperHessenberg( zeros( size(A[s]) ) ) for s in 1:length(A) ]

        )

        new(A, V, H, Arnoldi)

    end

end
   

mutable struct TensorLanczos{T} <: TensorDecomposition{T}

    A::KroneckerMatrix{T} # Original matrix
    V::KroneckerMatrix{T} # Matrix representing basis of Krylov subspace
    H::KroneckerMatrix{T} # Upper Hessenberg matrix
    orthonormalization::Type{Lanczos}

    function TensorLanczos{T}(A::KroneckerMatrix{T}) where T<:AbstractFloat

        V = KroneckerMatrix{T}(dimensions(A))
        H = KroneckerMatrix{T}(

            [ sparse(SymTridiagonal( zeros( size(A[s]) ) ))  for s in 1:length(A) ]

        )

        new(A, V, H, Lanczos)

    end

end


function update_subdiagonals!(H::AbstractMatrix{T}, k::Int, β::T) where T<:AbstractFloat

    indices = [CartesianIndex(k + 1, k), CartesianIndex(k, k + 1)]

    H[indices] .= β

end

function orthonormal_basis_vector!(arnoldi::Arnoldi{T}, k::Int) where T<:AbstractFloat

    v = Array{T}(undef, (size(arnoldi.A, 1)))

    LinearAlgebra.mul!(v, arnoldi.A, @view(arnoldi.V[:, k])) 

    for i = 1:k

        arnoldi.H[i, k] = dot(v, @view(arnoldi.V[:, i]))

        v .-= arnoldi.H[i, k] * @view(arnoldi.V[:, i])
    end

    arnoldi.H[k + 1, k] = LinearAlgebra.norm(v)

    arnoldi.V[:, k + 1] = v .* inv(arnoldi.H[k + 1, k])

end

function orthonormal_basis_vector!(lanczos::Lanczos{T}, k::Int) where T<:AbstractFloat

    n = order(lanczos)
    u = ones(n)

    # uₖ = Avₖ - βₖ₋₁vₖ₋₁
    LinearAlgebra.mul!(u, lanczos.A, lanczos.V[:, k])
    
    u -= lanczos.β .* lanczos.v

    # γₖ = <uₖ, vₖ>
    lanczos.H[k, k] = dot( u, @view(lanczos.V[:, k]) )

    # v̂ₖ₊₁ = uₖ - γₖvₖ
    v = u - (lanczos.H[k, k] .* @view(lanczos.V[:, k]))

    # βₖ = ||v̂ₖ₊₁||
    lanczos.β = LinearAlgebra.norm(v)

    # Update basis vector
    lanczos.β == 0.0 ? lanczos.V[:, k + 1] = zeros(n) : lanczos.V[:, k + 1] = inv(lanczos.β) .* v 

    # Set v to last vector in orthonormalization
    lanczos.v = @view(lanczos.V[:, k])

    # Update Jacobi matrix
    update_subdiagonals!(lanczos.H, k, lanczos.β)

end


function initial_orthonormalization!(t_decomp::TensorDecomposition{T}, b::KronProd{T}, decomp_type::Type{<:Decomposition}) where T<:AbstractFloat

    # This method performs the first orthonormalization step of both TensorArnoldi 
    # and TensorLanczos data structures.

    for s in 1:length(t_decomp)

        # Initialize d Arnoldi/Lanczos decompositions
        decomposition = decomp_type{T}(
            t_decomp.A.𝖳[s],
            t_decomp.V.𝖳[s],
            t_decomp.H.𝖳[s], 
            b[s])

        # First orthonormalization step for each of the coefficient matrices
        orthonormal_basis_vector!(decomposition, 1)

    end

end

function orthonormal_basis!(t_arnoldi::TensorArnoldi{T}, k::Int) where T<:AbstractFloat

    # This method performs not initial orthonormalization steps for the 
    # TensorArnoldi data structure.

    for s in 1:length(t_arnoldi)

        arnoldi = Arnoldi{T}(
            t_arnoldi.A.𝖳[s],
            t_arnoldi.V.𝖳[s],
            t_arnoldi.H.𝖳[s]) 

        orthonormal_basis_vector!(arnoldi, k)

    end

end

function orthonormal_basis!(t_lanczos::TensorLanczos{T}, k::Int) where T<:AbstractFloat

    # This method performs not initial orthonormalization steps for the 
    # TensorLanczos data structure.

    for s in 1:length(t_lanczos)

        lanczos = Lanczos{T}(
            t_lanczos.A.𝖳[s],
            t_lanczos.V.𝖳[s],
            t_lanczos.H.𝖳[s], 
            k)

        orthonormal_basis_vector!(lanczos, k)


    end

end

function lanczos_algorithm(A::AbstractMatrix{T}, b::AbstractVector{T}, k::Int) where T<:AbstractFloat

    n = size(A, 1)

    lanczos = Lanczos{T}(A, zeros(n, k), zeros(k, k), b)

    for j in 1:k-1

        orthonormal_basis_vector!(lanczos, j)

    end

    return lanczos

end

function isorthonormal(V::AbstractMatrix{T}, k::Int)::Bool where T<:AbstractFloat

    result = zeros(k, k)

    decomposition_basis = @view(V[:, 1:k])

    LinearAlgebra.mul!(result, transpose(decomposition_basis), decomposition_basis)

    return result ≈ I(k)

end

function isorthonormal(decomposition::Decomposition{T}, k::Int)::Bool where T<:AbstractFloat

    return isorthonormal(decomposition.V, k)

end

function isorthonormal(tensor_decomp::TensorDecomposition{T}, k::Int)::Bool where T<:AbstractFloat

    boolean = true

    for s in 1:length(tensor_decomp)

        boolean = boolean && isorthonormal(tensor_decomp.V[s], k)

    end

    return boolean
end

