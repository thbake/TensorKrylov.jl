export VectorCollection, MatrixCollection, KroneckerMatrix
export ConvDiff, EigValMat, Laplace, MatrixGallery, RandSPD
export assemble_matrix, kroneckervectorize,  kronproddot, kronprodnorm, 
       kth_columns, mul!, nentries, principal_minors,  randkronmat, size, 
       trikronmat 

abstract type VectorCollection{T} end
abstract type MatrixCollection{T} <: VectorCollection{T} end

abstract type MatrixGallery{T} end
struct Laplace{T}   <: MatrixGallery{T} end 
struct ConvDiff{T}  <: MatrixGallery{T} end
struct EigValMat{T} <: MatrixGallery{T} end
struct RandSPD{T}   <: MatrixGallery{T} end

const Core{T} = Vector{Vector{Vector{Vector{T}}}}


# We want to generate an abstract notion of structures that can be represented as Kronecker products
# or sums thereof, since all of these can be represented as vectors of abstract arrays
# or matrices

# Basic functions for VectorCollection types
# ==========================================
Base.length(collection::VectorCollection{T})           where T = length(collection.𝖳)

Base.getindex(collection::VectorCollection{T}, i::Int) where T = collection.𝖳[i]

Base.eachindex(collection::VectorCollection{T})        where T = eachindex(collection.𝖳)

Base.first(collection::VectorCollection{T})            where T = first(collection.𝖳)

Base.last(collection::VectorCollection{T})             where T = last(collection.𝖳)

function Base.getindex(collection::VectorCollection{T}, multiindex::Matrix{<:Int}) where T

    # Implement multiindexing of KroneckerProduct structures. To be precise
    # index a KroneckerProduct structure with a pair of multiindices ℑ₁, ℑ₂.
    d = size(multiindex, 1)

    length(collection) == d || throw(BoundsError(collection, multiindex))
    
    entries = zeros(d)

    for s = 1:length(collection)

        k, l = multiindex[s, :]
        
        entries[s] = collection[s][k, l]  

    end

    return entries

end

function Base.setindex!(collection::VectorCollection{T}, M::Matrix{T}, i::Int) where T

    1 <= i <= length(collection.𝖳) || throw(BoundsError(collection, i))

    collection.𝖳[i] = M

end


function dimensions(collection::VectorCollection{T})::Array{Int} where T
    
    factor_dimensions = Array{Int}(undef, length(collection))

    for s = 1:length(collection)
        
        factor_dimensions[s] = size(collection[s], 1)

    end

    return factor_dimensions

end

nentries(collection::VectorCollection{T}) where T = prod(dimensions(collection))

function assemble_matrix(n::Int, ::Type{Laplace{T}}) where T

    h  = inv(n + 1)
    Aₛ = inv(h^2) * sparse(SymTridiagonal(2ones(n), -ones(n)))

    return Aₛ

end

function assemble_matrix(n::Int, ::Type{ConvDiff{T}}, c::AbstractFloat = 10.0)  where T

    h  = inv(n + 1)
    L  =     sparse( inv(h^2) .* SymTridiagonal( 2ones(n), -ones(n - 1)) )
    Aₛ = L + sparse( (c * inv(4 * h)) .* diagm(-1 => ones(n-1), 0 => 3ones(n), 1 => -5ones(n - 1), 2 => ones(n - 2)) )

    return Aₛ

end

assemble_matrix(eigenvalues::Vector, ::Type{EigValMat{T}}) where T= diagm(eigenvalues)


function assemble_matrix(n::Int, ::Type{RandSPD{T}}) where T

    R = rand(n, n)

    return  Symmetric(R'R, :L)

end


    
struct KroneckerMatrix{T} <: MatrixCollection{T}
    
    𝖳           ::Vector{<:AbstractMatrix{T}} # We only store the d matrices explicitly in a vector.
    matrix_class::Type{<:MatrixGallery{T}}

    function KroneckerMatrix{T}(Aₛ::Vector{<:AbstractMatrix{T}}) where T 

        new(Aₛ, MatrixGallery{T})

    end

    function KroneckerMatrix{T}(Aₛ::Vector{<:AbstractMatrix{T}}, class::Type{<:MatrixGallery{T}}) where T# Constructor with vector of matrix coefficients

        new(Aₛ, class)

    end

    function KroneckerMatrix{T}(orders::Array{Int}) where T

        new([ zeros(orders[s], orders[s]) for s in 1:length(orders) ], MatrixGallery{T})

    end

    function KroneckerMatrix{T}(d::Int, n::Int, matrix_class::Type{<:MatrixGallery{T}}) where T

        A = assemble_matrix(n, matrix_class)
        KroneckerMatrix{T}( [ A for _ in 1:d ], matrix_class)

    end

    function KroneckerMatrix{T}(d::Int, eigenvalues, class::Type{EigValMat{T}}) where T

        A = assemble_matrix(eigenvalues, class)
        KroneckerMatrix{T}( [ A for _ in 1:d ], class)

    end

end

function Base.show(io::IO, A::KroneckerMatrix{T}) where T

    d = length(A)
    n = size(A[1], 1)
    println("Kronecker sum of order d = ", d, " with coefficient matrices of order n = ", n)
    println(typeof(A[1]))

end


function randkronmat(orders::Array{Int})

        return KroneckerMatrix{Float64}([ rand(order, order) for order in orders ])

end


function trikronmat(orders::Array{Int})

    return KroneckerMatrix{Float64}([ sparse(Tridiagonal( -ones(n - 1), 2ones(n), -ones(n - 1)))  for n in orders ])

end

function Base.size(KM::KroneckerMatrix{T})::Array{Tuple{Int, Int}} where T

    # Return size of each KroneckerMatrix element
    
    factor_sizes = Array{Tuple{Int, Int}}(undef, length(KM))

    for s = 1:length(KM)
        
        factor_sizes[s] = size(KM[s])

    end

    return factor_sizes
end

function kronproddot(v::AbstractArray{<:AbstractArray{T}}) where T

    return prod( dot(v[s], v[s]) for s in 1:length(v) ) 

end

function kronprodnorm(v::AbstractArray{<:AbstractArray{T}}) where T

    return sqrt( kronproddot(v) )

end

function principal_minors(v::AbstractArray{<:AbstractArray{T}}, i::Int) where T

    return [ @view(v[s][1:i]) for s in 1:length(v)]

end

function principal_minors(KM::KroneckerMatrix{T}, i::Int) where T

    return KroneckerMatrix{T}( [ @view(KM[s][1:i, 1:i]) for s in 1:length(KM)] )

end

function principal_minors(KM::KroneckerMatrix{T}, i::Int, j::Int) where T

    return KroneckerMatrix{T}( [ @view(KM[s][1:i, 1:j]) for s in 1:length(KM)] )

end

function principal_minors(x::ktensor, i::Int) 

    return ktensor(x.lambda, [ (x.fmat[s][1:i, :]) for s in 1:ndims(x) ] )

end

function Base.getindex(KM::KroneckerMatrix, i::Int, j::Int)

    # Return the entry (i, j) of all d coefficient matrices
    return [ KM[s][i, j] for s in 1:length(KM) ]

end

function kth_rows(KM::KroneckerMatrix, k::Int)

    return [ @view(KM[s][k, :]) for s in 1:length(KM) ]

end

function kth_columns(KM::KroneckerMatrix, k::Int)

    return [ @view(KM[s][:, k]) for s in 1:length(KM) ]

end

function kth_columns(A, k::Int)

    return [ @view(A[s][:, k]) for s in 1:length(A) ]

end

# Linear algebra for Kronecker matrices
LinearAlgebra.adjoint(A::KroneckerMatrix)   = adjoint.(A.𝖳)
LinearAlgebra.transpose(A::KroneckerMatrix) = transpose.(A.𝖳)

# Additional functionality for Kruskal tensors
# ============================================
function Base.getindex(CP::ktensor, i::Int)

    # Returns a vector containing the i-th column of each factor matrix of CP.

    return [ @view(CP.fmat[s][:, i]) for s = 1:ndims(CP) ]

end

function LinearAlgebra.mul!(result::Vector{<:AbstractMatrix{T}}, y::Vector{<:AbstractVector{T}}, x::ktensor) where T 

    # Compute product between elementary tensor and factor matrices of Kruskal tensor.
    nᵢ = ndims(x)

   for s = 1:nᵢ

       # Result is vector of row vectors
       result[s] = transpose(y[s]) * x.fmat[s]

   end

end

function kroneckervectorize(x::ktensor)

    N    = prod(size(x))
    vecx = zeros(N)

    for i in 1:ncomponents(x) 

        tmp = @view(x.fmat[end][:, i])

        for j in ndims(x) - 1 : - 1 : 1

            tmp = kron(tmp, @view(x.fmat[j][:, i]))

        end

        vecx += tmp

    end

    return vecx

end

function mul!(
        result::Vector{<:AbstractMatrix{T}},
        x::Vector{<:AbstractVector{T}},
        A::Vector{<:AbstractMatrix{T}}) where T 

    # Compute product between vector of collection of row vectors and matrices.
    nᵢ = length(result)

    for s = 1:nᵢ

        result[s] = transpose(x[s]) * A[s]

    end

end

struct TTCore{T} 

    core_tensor::Core{T}

    function TTCore{T}(orders::AbstractVector{Int}) where T

        @assert length(orders) == 4

        new(zeros(orders...))

    end

    function TTCore{T}(core::Core{T}) where T

        @assert length(core) == 4

        new(core_tensor)

    end

end

function Base.length(core::TTCore{T}) where T

    return length(core.core_tensor)

end

function Base.size(core::TTCore{T}) where T

    return [ size(core[s]) for s in 1:length(core) ]

end

