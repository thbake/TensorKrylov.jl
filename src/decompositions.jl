export Arnoldi, Lanczos, TensorArnoldi, TensorLanczos, multiple_arnoldi!, multiple_lanczos!, arnoldi_step!, lanczos_step!

const MatrixView{T}    = AbstractMatrix{T}
const KroneckerProd{T} = AbstractArray{<:AbstractArray{T}}

abstract type Decomposition{T<:AbstractFloat}       end
abstract type TensorDecomposition{T<:AbstractFloat} end

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

function update_subdiagonals!(T::AbstractMatrix{U}, j::Int, β::U) where U<:AbstractFloat

    indices = [CartesianIndex(j + 1, j), CartesianIndex(j, j + 1)]

    T[indices] .= β

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


        # Initialization: perform orthonormalization for second basis vector

        n = size(A, 1)

        V[:, 1] = inv( LinearAlgebra.norm(b) ) .* b

        u = zeros(n) 

        LinearAlgebra.mul!(u, A, @view(V[:, 1]))

        v = zeros(n)

        v = u - dot(u, @view(V[:, 1])) .* @view(V[:, 1])

        β = LinearAlgebra.norm(v)

        β == 0.0 ? V[:, 2] = zeros(n) : V[:, 2] = v .* inv(β)

        update_subdiagonals!(T, 1, β)

        new(A, V, T)

    end

end

function lanczos_step!(lanczos::Lanczos{U}, j::Int) where U<:AbstractFloat

    # First step is performed in initialization
    @assert j >= 2

    n = order(lanczos)
    u = zeros(n)

    LinearAlgebra.mul!(u, lanczos.A, @view(lanczos.V[:, j]))

    u -= lanczos.T[j, j-1] .* @view(lanczos.V[:, j - 1])

    lanczos.T[j, j] = dot( u, @view(lanczos.V[:, j]) )

    v = u - lanczos.T[j, j] .* @view(lanczos.V[:, j])

    β = LinearAlgebra.norm(v)

    # Update basis vector
    β == 0.0 ? lanczos.V[:, j + 1] = zeros(n) : lanczos.V[:, j + 1] = inv(β) .* v 
    #if β == 0.0

        #@info "I'm 0"
        #lanczos.V[:, j] = zeros()

    # Update Jacobi matrix
    update_subdiagonals!(lanczos.T, j, β)

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


mutable struct TensorLanczos{U} <: TensorDecomposition{U}

    A::KroneckerMatrix{U} # Original matrix
    V::KroneckerMatrix{U} # Matrix representing basis of Krylov subspace
    T::KroneckerMatrix{U} # Upper Hessenberg matrix

    function TensorLanczos{U}(A::KroneckerMatrix{U}) where U<:AbstractFloat

        V = KroneckerMatrix{U}(dimensions(A))
        T = KroneckerMatrix{U}(dimensions(A))

        new(A, V, T)

    end

end


function multiple_lanczos!(t_lanczos::TensorLanczos{U}, b::KroneckerProd{U}, j::Int) where U<:AbstractFloat

    for s in 1:length(t_lanczos)

        lanczos = Lanczos{U}(
                t_lanczos.A.𝖳[s], 
                t_lanczos.V.𝖳[s],
                t_lanczos.T.𝖳[s], 
                b[s])

        lanczos_step!(lanczos, j)

    end

end
