export Arnoldi

mutable struct Arnoldi{T<:AbstractFloat} # Stores Krylov basis and upper Hessenberg matrix
    const A::Matrix{T} # Original matrix
    V::Matrix{T} # Matrix representing basis of Krylov subspace
    H::Matrix{T} # Upper Hessenberg matrix

    function Arnoldi(A::Matrix{T}, order::Int) where T<:AbstractFloat
        new{T}(
            A, 
            zeros(T, size(A, 1), order + 1), # Initialize Krylov basis
            UpperHessenberg(
                zeros(T, order + 1, order)
            )::UpperHessenberg       # Initialize upper Hessenberg matrix
        )  
    end
end
