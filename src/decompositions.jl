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

    function Lanczos{T}(
                A::MatrixView{T},
                V::MatrixView{T},
                H::MatrixView{T},
                b::AbstractArray{T}) where T<:AbstractFloat


        # Initialization: perform orthonormalization for second basis vector

        n = size(A, 1)

        V[:, 1] = inv( LinearAlgebra.norm(b) ) .* b

        u = zeros(n) 

        LinearAlgebra.mul!(u, A, @view(V[:, 1]))

        v = zeros(n)

        v = u - dot(u, @view(V[:, 1])) .* @view(V[:, 1])

        β = LinearAlgebra.norm(v)

        β == 0.0 ? V[:, 2] = zeros(n) : V[:, 2] = v .* inv(β)

        update_subdiagonals!(H, 1, β)

        new(A, V, H)

    end

end


mutable struct TensorArnoldi{T} <: TensorDecomposition{T}

    A::KroneckerMatrix{T} # Original matrix
    V::KroneckerMatrix{T} # Matrix representing basis of Krylov subspace
    H::KroneckerMatrix{T} # Upper Hessenberg matrix

    function TensorArnoldi{T}(A::KroneckerMatrix{T}) where T<:AbstractFloat

        V = KroneckerMatrix{T}(dimensions(A))
        H = KroneckerMatrix{T}(

            [ UpperHessenberg( zeros( size(A[s]) ) ) for s in 1:length(A) ]

        )

        new(A, V, H)

    end

end
   

mutable struct TensorLanczos{T} <: TensorDecomposition{T}

    A::KroneckerMatrix{T} # Original matrix
    V::KroneckerMatrix{T} # Matrix representing basis of Krylov subspace
    H::KroneckerMatrix{T} # Upper Hessenberg matrix

    function TensorLanczos{T}(A::KroneckerMatrix{T}) where T<:AbstractFloat

        V = KroneckerMatrix{T}(dimensions(A))
        H = KroneckerMatrix{T}(

           [ Tridiagonal( zeros( size(A[s]) ) ) for s in 1:length(A) ]

        )

        new(A, V, H)

    end

end

function update_subdiagonals!(H::AbstractMatrix{T}, j::Int, β::T) where T<:AbstractFloat

    indices = [CartesianIndex(j + 1, j), CartesianIndex(j, j + 1)]

    H[indices] .= β

end

function orthonormal_basis_vector!(arnoldi::Arnoldi{T}, j::Int) where T<:AbstractFloat

    v = Array{T}(undef, (size(arnoldi.A, 1)))

    LinearAlgebra.mul!(v, arnoldi.A, @view(arnoldi.V[:, j])) 

    for i = 1:j

        arnoldi.H[i, j] = dot(v, @view(arnoldi.V[:, i]))

        v .-= arnoldi.H[i, j] * @view(arnoldi.V[:, i])
    end

    arnoldi.H[j + 1, j] = LinearAlgebra.norm(v)

    arnoldi.V[:, j + 1] = v .* inv(arnoldi.H[j + 1, j])

end

function orthonormal_basis_vector!(lanczos::Lanczos{T}, j::Int) where T<:AbstractFloat

    # First step is performed in initialization
    @assert j >= 2

    n = order(lanczos)
    u = zeros(n)

    LinearAlgebra.mul!(u, lanczos.A, @view(lanczos.V[:, j]))

    u -= lanczos.H[j, j-1] .* @view(lanczos.V[:, j - 1])

    lanczos.H[j, j] = dot( u, @view(lanczos.V[:, j]) )

    v = u - lanczos.H[j, j] .* @view(lanczos.V[:, j])

    β = LinearAlgebra.norm(v)

    # Update basis vector
    β == 0.0 ? lanczos.V[:, j + 1] = zeros(n) : lanczos.V[:, j + 1] = inv(β) .* v 

    # Update Jacobi matrix
    update_subdiagonals!(lanczos.H, j, β)

end

function orthonormal_basis!(t_decomp::TensorDecomposition{T}, b::KroneckerProd{T}, j::Int, decomp_type::Type{<:Decomposition}) where T<:AbstractFloat

    for s in 1:length(t_decomp)

        decomposition = decomp_type{T}(
                t_decomp.A.𝖳[s], 
                t_decomp.V.𝖳[s],
                t_decomp.H.𝖳[s], 
                b[s])

        orthonormal_basis_vector!(decomposition, j)

    end

end
