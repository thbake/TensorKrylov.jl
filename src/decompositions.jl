export Arnoldi, Lanczos, TensorArnoldi, TensorLanczos
export orthonormal_basis!, orthonormal_basis_vector!

const MatrixView{T}    = AbstractMatrix{T}
const KroneckerProd{T} = AbstractArray{<:AbstractArray{T}}

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

    #function Arnoldi{T}(A::MatrixView{T}, b::AbstractArray{T}) where T<:AbstractFloat

    #end

    function Arnoldi{T}(
                A::MatrixView{T},
                V::MatrixView{T},
                H::MatrixView{T},
                b::AbstractArray{T}) where T<:AbstractFloat


        V[:, 1] = inv( LinearAlgebra.norm(b) ) .* b

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


        # Initialization: perform orthonormalization for second basis vector

        β = 0.0

        V[:, 1] = inv( LinearAlgebra.norm(b) ) .* b

        v = zeros(size(A, 1))

        new(A, V, H, β, v)

    end

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

            [ sparse(Tridiagonal( zeros( size(A[s]) ) ) ) for s in 1:length(A) ]

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
    u = zeros(n)

    #LinearAlgebra.mul!(u, lanczos.A, @view(lanczos.V[:, j]))
    LinearAlgebra.mul!(u, lanczos.A, @view(lanczos.V[:, k]))

    u -= lanczos.β .* lanczos.v
    #u -= lanczos.H[j, j-1] .* @view(lanczos.V[:, j - 1])

    #lanczos.H[j, j] = dot( u, @view(lanczos.V[:, j]) )
    lanczos.H[k, k] = dot( u, @view(lanczos.V[:, k]) )

    #v = u - lanczos.H[j, j] .* @view(lanczos.V[:, j])
    v = u - lanczos.H[k, k] .* @view(lanczos.V[:, k])

    lanczos.β = LinearAlgebra.norm(v)

    # Update basis vector
    lanczos.β == 0.0 ? lanczos.V[:, k + 1] = zeros(n) : lanczos.V[:, k + 1] = inv(lanczos.β) .* v 

    lanczos.v = @view(lanczos.V[:, k])

    # Update Jacobi matrix
    update_subdiagonals!(lanczos.H, k, lanczos.β)

end

function orthonormal_basis!(t_decomp::TensorDecomposition{T}, b::KroneckerProd{T}, k::Int, decomp_type::Type{<:Decomposition}) where T<:AbstractFloat

    for s in 1:length(t_decomp)

        decomposition = decomp_type{T}(
                t_decomp.A.𝖳[s], 
                t_decomp.V.𝖳[s],
                t_decomp.H.𝖳[s], 
                b[s])

        orthonormal_basis_vector!(decomposition, k)

    end

end

function lanczos_basis!(lanczos::Lanczos{T}, k::Int) where T<:AbstractFloat

    for i in 1:k-1

        orthonormal_basis_vector!(lanczos, i)

    end

end

function arnoldi_basis!(arnoldi::Arnoldi{T}, k::Int) where T<:AbstractFloat

    for i in 1:k-1

        orthonormal_basis_vector!(lanczos, i)

    end

end
