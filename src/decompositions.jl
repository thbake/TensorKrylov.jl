export TensorArnoldi, Arnoldi, TensorLanczos, multiple_arnoldi!

const MatrixView{T}    = AbstractMatrix{T}
const KroneckerProd{T} = AbstractArray{<:AbstractArray{T}}

abstract type Decomposition{T<:AbstractFloat}       end
abstract type TensorDecomposition{T<:AbstractFloat} end

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

function arnoldi_step!(arnoldi::Arnoldi{T}, j::Int) where T<:AbstractFloat

    v = Array{T}(undef, (size(arnoldi.A, 1)))

    LinearAlgebra.mul!(v, arnoldi.A, @view(arnoldi.V[:, j])) 

    for i = 1:j

        arnoldi.H[i, j] = dot(v, @view(arnoldi.V[:, i]))

        v .-= arnoldi.H[i, j] * @view(arnoldi.V[:, i])
    end

    arnoldi.H[j + 1, j] = LinearAlgebra.norm(v)

    arnoldi.V[:, j + 1] = v .* inv(arnoldi.H[j + 1, j])

end

mutable struct Lanczos{U} <: Decomposition{U}

    A::MatrixView{U}
    V::MatrixView{U}
    T::MatrixView{U}

    function Lanczos{U}(
                A::MatrixView{U},
                V::MatrixView{U},
                T::MatrixView{U},
                b::AbstractArray{U}) where U<:AbstractFloat


        V[:, 1] = inv( LinearAlgebra.norm(b) ) .* b

        new(A, V, T)

    end

end

function lanczos_step!(lanczos::Lanczos{U}, j::Int) where U<:AbstractFloat

    u = lanczos.A * @view(lanczos.V[:, j]) - lanczos.T[j-1, j] .* lanczos.V[:, j-1]

    lanczos.T[j,j] = dot(u, lanczos.V[:, j])

    v = u - lanczos.T[j,j] .* @view(lanczos.V[:, j])

    β = norm(v)

    if β == 0.0

        lanczos.V[:, j + 1] = zeros(size(lanczos.V, 1))

    else
        lanczos.V[:, j + 1] = v .* inv(β)

end


mutable struct TensorArnoldi{T} <: TensorDecomposition{T}

    A::KroneckerMatrix{T} # Original matrix
    V::KroneckerMatrix{T} # Matrix representing basis of Krylov subspace
    H::KroneckerMatrix{T} # Upper Hessenberg matrix

    function TensorArnoldi{T}(A::KroneckerMatrix{T}) where T<:AbstractFloat

        V = KroneckerMatrix{T}(dimensions(A))
        H = KroneckerMatrix{T}(dimensions(A))

        new(A, V, H)

    end

end
   

function multiple_arnoldi!(t_arnoldi::TensorArnoldi{T}, b::KroneckerProd{T}, j::Int) where T<:AbstractFloat

    for s in 1:length(t_arnoldi)

        arnoldi = Arnoldi{T}(
                t_arnoldi.A.𝖳[s], 
                t_arnoldi.V.𝖳[s],
                t_arnoldi.H.𝖳[s], 
                b[s])

        arnoldi_step!(arnoldi, j)

    end
end


mutable struct TensorLanczos{T<:AbstractFloat}

    A::KroneckerMatrix{T} # Original matrix
    V::KroneckerMatrix{T} # Matrix representing basis of Krylov subspace
    H::KroneckerMatrix{T} # Upper Hessenberg matrix

    function TensorLanczos{T}(A::KroneckerMatrix{T}, b::Vector{Vector{T}}) where T<:AbstractFloat

        d = length(A)
        
        V = KroneckerMatrix{T}(dimensions(A))
        H = KroneckerMatrix{T}(dimensions(A))

        for s = 1:d

            V[s][:, 1] = inv( LinearAlgebra.norm(b[s]) ) .* b[s]

        end

        new(A, V, H)

    end

end
