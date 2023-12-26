export Arnoldi, Lanczos, TensorDecomposition, TensorArnoldi, TensorLanczos, 
       TensorLanczosReorth, LanczosReorth, MGS, LanczosUnion

const MatrixView{T}    = AbstractMatrix{T}

abstract type Decomposition{T}       end
abstract type TensorDecomposition{T} end
abstract type CheapOrthogonalityLoss end

struct MGS end

function order(decomposition::Decomposition)

    return size(decomposition.A, 1)

end

function Base.length(tensor_decomp::TensorDecomposition) 

    return length(tensor_decomp.A)

end

function Base.getindex(tensordecomp::TensorDecomposition{T}, i::Int) where T<:AbstractFloat

    1 <= i <= length(tensordecomp) || throw(BoundsError(tensordecomp, i))

    return tensordecomp.A[i], tensordecomp.V[i], tensordecomp.H[i]

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

mutable struct LanczosReorth{T} <: Decomposition{T}
    A::MatrixView{T}
    V::MatrixView{T}
    H::MatrixView{T}
    β::T                # Last off-diagonal coefficient in H
    v::AbstractArray{T} # Last basis vector

    function LanczosReorth{T}(
                A::MatrixView{T},
                V::MatrixView{T},
                H::MatrixView{T},
                b::AbstractArray{T}) where T<:AbstractFloat
                
        lanczos = Lanczos{T}(A, V, H, b)

        new(lanczos.A, lanczos.V, lanczos.H, lanczos.β, lanczos.v)

    end

    function LanczosReorth{T}(
                A::MatrixView{T},
                V::MatrixView{T},
                H::MatrixView{T},
                k::Int) where T<:AbstractFloat

        lanczos = Lanczos{T}(A, V, H, k)

        new(lanczos.A, lanczos.V, lanczos.H, lanczos.β, lanczos.v)

    end

end

function initialize_decomp!(V::MatrixView{T}, b::AbstractArray{T}) where T<:AbstractFloat

        # Normalize first basis vector

        V[:, 1] = inv( norm(b) ) .* b

end

mutable struct TensorArnoldi{T} <: TensorDecomposition{T}

    A::KroneckerMatrix{T} # Original matrix
    V::KroneckerMatrix{T} # Matrix representing basis of Krylov subspace
    H::KroneckerMatrix{T} # Upper Hessenberg matrix
    orthonormalization::Type{Arnoldi{T}}

    function TensorArnoldi{T}(A::KroneckerMatrix{T}) where T<:AbstractFloat

        V = KroneckerMatrix{T}(dimensions(A))
        H = KroneckerMatrix{T}(

            [ UpperHessenberg( zeros( size(A[s]) ) ) for s in 1:length(A) ]

        )

        new(A, V, H, Arnoldi{T})

    end

end
   

mutable struct TensorLanczos{T} <: TensorDecomposition{T}

    A::KroneckerMatrix{T} # Original matrix
    V::KroneckerMatrix{T} # Matrix representing basis of Krylov subspace
    H::KroneckerMatrix{T} # Upper Hessenberg matrix
    orthonormalization::Type{Lanczos{T}}

    function TensorLanczos{T}(A::KroneckerMatrix{T}) where T<:AbstractFloat

        V = KroneckerMatrix{T}(dimensions(A))
        H = KroneckerMatrix{T}(

            [ sparse(SymTridiagonal( zeros( size(A[s]) ) ))  for s in 1:length(A) ]

        )

        new(A, V, H, Lanczos{T})

    end

end

mutable struct TensorLanczosReorth{T} <: TensorDecomposition{T}

    A::KroneckerMatrix{T} # Original matrix
    V::KroneckerMatrix{T} # Matrix representing basis of Krylov subspace
    H::KroneckerMatrix{T} # Upper Hessenberg matrix
    orthonormalization::Type{LanczosReorth}

    function TensorLanczosReorth{T}(A::KroneckerMatrix{T}) where T<:AbstractFloat

        V = KroneckerMatrix{T}(dimensions(A))
        H = KroneckerMatrix{T}(

            [ sparse(SymTridiagonal( zeros( size(A[s]) ) ))  for s in 1:length(A) ]

        )

        new(A, V, H, LanczosReorth{T})

    end

end

const LanczosUnion{T} = Union{Type{TensorLanczos{T}}, Type{TensorLanczosReorth{T}}}

function update_subdiagonals!(H::AbstractMatrix{T}, k::Int, β::T) where T<:AbstractFloat

    indices = [CartesianIndex(k + 1, k), CartesianIndex(k, k + 1)]

    H[indices] .= β

end

#function orthonormal_basis_vector!(decomposition::Decomposition{T}, k::Int, ::MGS) where T<:AbstractFloat
#
#    v = zeros( size(decomposition.A, 1) )
#
#    mul!(v, decomposition.A, @view decomposition.V[:, k])
#
#    for i = 1:k
#
#        decomposition.H[i, k] = dot(v, @view decomposition.V[:, i])
#
#        v .-= decomposition.H[i, k] .* @view decomposition.V[:, i]
#
#    end
#
#    decomposition.H[k + 1, k] = norm(v)
#
#    decomposition.V[:, k + 1] = v .* inv(decomposition.H[k + 1, k])
#
#end

