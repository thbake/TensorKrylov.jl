export Arnoldis

mutable struct Arnoldis{T<:AbstractFloat}

    A::KroneckerMatrix{T} # Original matrix
    V::KroneckerMatrix{T} # Matrix representing basis of Krylov subspace
    H::KroneckerMatrix{T} # Upper Hessenberg matrix

    function Arnoldis{T}(A::KroneckerMatrix{T}, b::Vector{Vector{T}}) where T<:AbstractFloat

        d = length(A)
        
        V = KroneckerMatrix{T}(dimensions(A))
        H = KroneckerMatrix{T}(dimensions(A))

        for s = 1:d

            V[s][:, 1] = inv( LinearAlgebra.norm(b[s]) ) .* b[s]

        end

        new(A, V, H)

    end

end
   

function arnoldi_step!(arnoldis::Arnoldis{T}, index::Int) where T<:AbstractFloat

    for s in 1:length(arnoldis.A)

        v = Array{Float64}(undef, ( size(arnoldis.A[s], 1) ))

        mul!(v, arnoldis.A[s], @views(arnoldis.V[s][:, index])) 

        for i = 1:index

            arnoldis.H[s][i, index] = dot(v, @views(arnoldis.V[s][:, i]))

            v .-= arnoldis.H[s][i, index] * @views(arnoldis.V[s][:, i])

        end

        arnoldis.V[s][:, index + 1] = v .* inv(arnoldis.H[s][index + 1, index])

    end

end
