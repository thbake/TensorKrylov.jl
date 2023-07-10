export Arnoldi

mutable struct Arnoldi{T<:AbstractFloat}

    A::KroneckerMatrix{T} # Original matrix
    V::KroneckerMatrix{T} # Matrix representing basis of Krylov subspace
    H::KroneckerMatrix{T} # Upper Hessenberg matrix

    function Arnoldi{T}(A::KroneckerMatrix{T}, b::Vector{Vector{T}}) where T<:AbstractFloat

        d = length(A)
        
        V = KroneckerMatrix{T}(dimensions(A))
        H = KroneckerMatrix{T}(dimensions(A))

        for s = 1:d

            V[s][:, 1] = inv( LinearAlgebra.norm(b[s]) ) .* b[s]

        end

        new(A, V, H)

    end

end
   

function arnoldi_step!(arnoldi::Arnoldi{T}, j::Int) where T<:AbstractFloat

    for s in 1:length(arnoldi.A)

        v = Array{Float64}(undef, (size(arnoldi.A[s], 1)))

        LinearAlgebra.mul!(v, arnoldi.A[s], @view(arnoldi.V[s][:, j])) 

        for i = 1:j

            arnoldi.H[s][i, j] = dot(v, @view(arnoldi.V[s][:, i]))

            v .-= arnoldi.H[s][i, j] * @view(arnoldi.V[s][:, i])
        end

        arnoldi.H[s][j + 1, j] = LinearAlgebra.norm(v)

        arnoldi.V[s][:, j + 1] = v .* inv(arnoldi.H[s][j + 1, j])

    end

end
