export Arnoldi, Lanczos, TensorDecomposition, TensorArnoldi, TensorLanczos, 
       TensorLanczosReorth, LanczosReorth,  LanczosUnion

abstract type Decomposition{matT, T}          end
abstract type TensorDecomposition{matT, T, U} end
abstract type CheapOrthogonalityLoss end

function order(decomposition::Decomposition)

    return size(decomposition.A, 1)

end

function Base.length(tensor_decomp::TensorDecomposition) 

    return length(tensor_decomp.A)

end

function Base.getindex(tensordecomp::TensorDecomposition, i::Int) 

    1 <= i <= length(tensordecomp) || throw(BoundsError(tensordecomp, i))

    return tensordecomp.A[i], tensordecomp.V[i], tensordecomp.H[i]

end

mutable struct Arnoldi{matT, T} <: Decomposition{matT, T}

    A::matT
    V::Matrix{T}
    H::Matrix{T}

    function Arnoldi(A::matT, V::Matrix{T}, H::Matrix{T}, b::Vector{T}) where {matT, T}

        initialize_decomp!(V, b)

        new{matT, T}(A, V, H)

    end

    function Arnoldi(A::matT, V::Matrix{T}, H::Matrix{T}) where {matT, T}

        new{matT, T}(A, V, H)

    end
    
    function Arnoldi(A::matT, V::Matrix{T}, H::Matrix{T}, ::Int) where {matT, T}

        Arnoldi(A, V, H)

    end
                       
end

mutable struct Lanczos{matT, T} <: Decomposition{matT, T}

    A::matT
    V::Matrix{T}
    H::Matrix{T}
    β::T         # Last off-diagonal coefficient in H
    v::Vector{T} # Last basis vector

    function Lanczos(A::matT, V::Matrix{T}, H::Matrix{T}, b::Vector{T}) where {matT, T}

        # β₀ = 0, v₀ = 0
        β = 0.0
        v = zeros(size(A, 1)) # vₖ₋₁

        initialize_decomp!(V, b)

        new{matT, T}(A, V, H, β, v)

    end

    function Lanczos(A::matT, V::Matrix{T}, H::Matrix{T}, k::Int) where {matT, T}

        β = H[k - 1, k]
        v = @view(V[:, k - 1])

        new{matT, T}(A, V, H, β, v)

    end

end

mutable struct LanczosReorth{matT, T} <: Decomposition{matT, T}
    A::matT
    V::Matrix{T}
    H::Matrix{T}
    β::T         # Last off-diagonal coefficient in H
    v::Vector{T} # Last basis vector

    function LanczosReorth(A::matT, V::Matrix{T}, H::Matrix{T}, b::Vector{T}) where {matT, T}

        lanczos = Lanczos(A, V, H, b)

        new{matT, T}(lanczos.A, lanczos.V, lanczos.H, lanczos.β, lanczos.v)

    end

    function LanczosReorth(A::matT, V::Matrix{T}, H::Matrix{T}, k::Int) where {matT, T}

        lanczos = Lanczos(A, V, H, k)

        new{matT, T}(lanczos.A, lanczos.V, lanczos.H, lanczos.β, lanczos.v)

    end

end

function initialize_decomp!(V::Matrix{T}, b::Vector{T}) where T

        # Normalize first basis vector

        V[:, 1] = inv( norm(b) ) .* b

end

mutable struct TensorArnoldi{matT, T, U} <: TensorDecomposition{matT, T, U}

    A::KronMat{ matT, U}      # Original matrix
    V::KronComp{Matrix{T}}    # Matrix representing basis of Krylov subspace
    H::KronMat{ Matrix{T}, U} # Upper Hessenberg matrix
    orthonormalization::Type{Arnoldi}

    function TensorArnoldi(A::KronMat{matT, U}) where {matT, U<:Instance}

        T = eltype(matT)
        V = KronComp{Matrix{T}}(dimensions(A)   )
        H = KronMat{Matrix{T}, U}(dimensions(A) .+1)

        new{matT, T, U}(A, V, H, Arnoldi)

    end

end
   

mutable struct TensorLanczos{matT, T, U} <: TensorDecomposition{matT, T, U}

    A::KronMat{matT, U}    # Original matrix
    V::KronComp{Matrix{T}} # Matrix representing basis of Krylov subspace
    H::KronMat{Matrix{T}, U}    # Upper Hessenberg matrix
    orthonormalization::Type{Lanczos}

    function TensorLanczos(A::KronMat{matT, U}) where {matT, U<:Instance}

        T = eltype(matT)
        V = KronComp{Matrix{T}}(dimensions(A)   )
        H = KronMat{Matrix{T}, U}(dimensions(A) .+ 1)

        new{matT, T, U}(A, V, H, Lanczos)

    end

end

mutable struct TensorLanczosReorth{matT, T, U} <: TensorDecomposition{matT, T, U}

    A::KronMat{matT, U}      # Original matrix
    V::KronComp{Matrix{T}}   # Matrix representing basis of Krylov subspace
    H::KronMat{Matrix{T}, U} # Upper Hessenberg matrix
    orthonormalization::Type{LanczosReorth}

    function TensorLanczosReorth(A::KronMat{matT, U}) where {matT, U<:Instance}

        T = eltype(matT)
        V = KronComp{Matrix{T}}(dimensions(A)   )
        H = KronMat{Matrix{T}, U}(dimensions(A) .+ 1)

        new{matT, T, U}(A, V, H, LanczosReorth)

    end

end

const LanczosUnion{matT, T, U} = Union{Type{TensorLanczos{matT, T, U}}, Type{TensorLanczosReorth{matT, T, U}}}

function update_subdiagonals!(H::Matrix{T}, k::Int, β::T) where T

    indices = [CartesianIndex(k + 1, k), CartesianIndex(k, k + 1)]

    H[indices] .= β

end
