export Arnoldi

abstract type Decomposition end 

mutable struct Arnoldi{T<:AbstractFloat} <: Decomposition # Stores Krylov basis and upper Hessenberg matrix
    A::Matrix{T} # Original matrix
    V::Matrix{T} # Matrix representing basis of Krylov subspace
    H::UpperHessenberg{T, Matrix{T}} # Upper Hessenberg matrix

    function Arnoldi(A::Matrix{T}, order::Int) where T<:AbstractFloat
        new{T}(
            A, 
            zeros(T, size(A, 1), order + 1), # Initialize Krylov basis
            UpperHessenberg(
                zeros(T, order + 1, order)
            )::UpperHessenberg       # Initialize upper Hessenberg matrix
        )  
    end

    function Arnoldi(A::Matrix{T}, b::Vector{T}) where T<:AbstractFloat

        V = zeros(size(A))

        V[:, 1] = inv(norm(b)) .* b # Already initialize first column of struct

        new{T}(A, V, UpperHessenberg( zeros(size(A)) ))

    end

end

function arnoldi_step!(arnoldi::Arnoldi{T}, index::Int) where T<:AbstractFloat

    v = Array{Float64}(undef, ( size(arnoldi.A, 1) ))

    mul!(v, arnoldi.A, @views(arnoldi.V[:, index])) 

    for i = 1:index

        arnoldi.H[i, index] = dot(v, @views(arnoldi.V[:, i]))

        v .-= arnoldi.H[i, index] * @views(arnoldi.V[:, i])

    end

    arnoldi.V[:, index + 1] = v .* inv(arnoldi.H[index + 1, index])

end
