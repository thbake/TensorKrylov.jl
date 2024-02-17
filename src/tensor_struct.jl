using MatrixDepot

export KroneckerMatrix, KruskalTensor
export ConvDiff, EigValMat, Laplace, LaplaceDense, MatrixGallery, RandSPD,
       MatrixDep

export Instance, SymInstance, NonSymInstance

export assemble_matrix, dimensions, kroneckervectorize,  kronproddot, 
       kronprodnorm, kth_columns, mul!, ncomponents, ndims, nentries, principal_minors,  
       randkronmat, size, trikronmat 

const KronStruct{T} = Vector{<:AbstractVecOrMat{T}}
const FMatrices{T}  = Vector{<:AbstractMatrix{T}}
const CoeffMat{matT} = Vector{matT}
const MatrixView{matT, T} = Array{SubArray{T, 2, matT, Tuple{UnitRange{Int}, UnitRange{Int}}, false}, 1}

abstract type MatrixGallery end
struct LaplaceDense <: MatrixGallery end
struct Laplace      <: MatrixGallery end 
struct ConvDiff     <: MatrixGallery end
struct EigValMat    <: MatrixGallery end
struct RandSPD      <: MatrixGallery end

struct MatrixDep{T}  <: MatrixGallery 

    matrix_id::String
    n        ::Int
    M        ::Matrix{T}

    function MatrixDep()

        new{Float64}("", 1, zeros(1,1))

    end

    function MatrixDep(matrix_id::String, n::Int)

        M = matrixdepot(matrix_id, n)

        T = eltype(M)

        new{T}(matrix_id, n, M)

    end
end

function assemble_matrix(n::Int, ::Type{LaplaceDense}) 

    h  = inv(n + 1)
    Aₛ = inv(h^2) * SymTridiagonal(2ones(n), -ones(n))

    return Aₛ

end

assemble_matrix(n::Int, ::Type{Laplace}) = sparse( assemble_matrix(n, LaplaceDense) )


function assemble_matrix(n::Int, ::Type{ConvDiff}, c::AbstractFloat = 10.0)  

    h  = inv(n + 1)
    L  =     sparse( inv(h^2) .* SymTridiagonal( 2ones(n), -ones(n - 1)) )
    Aₛ = L + sparse( (c * inv(4 * h)) .* diagm(-1 => ones(n-1), 0 => 3ones(n), 1 => -5ones(n - 1), 2 => ones(n - 2)) )

    return Aₛ

end

assemble_matrix(eigenvalues::Vector, ::Type{EigValMat}) = diagm(eigenvalues)


function assemble_matrix(n::Int, ::Type{RandSPD})

    R = rand(n, n)

    return  Symmetric(R'R, :L)

end

assemble_matrix(n::Int, ::Type{MatrixDep}, name::String) = matrixdepot(name, n)

abstract type Instance end
struct SymInstance    <: Instance end
struct NonSymInstance <: Instance end

# We want to generate an abstract notion of structures that can be represented as Kronecker products
# or sums thereof, since all of these can be represented as vectors of abstract arrays
# or matrices

abstract type AbstractKroneckerMatrix{matT} end

# Basic functions for KroneckerMatrix types
# ==========================================
Base.length(A::AbstractKroneckerMatrix)           = length(A.M)

Base.getindex(A::AbstractKroneckerMatrix, i::Int) = A.M[i]

Base.eachindex(A::AbstractKroneckerMatrix)        = eachindex(A.M)

function Base.getindex(A::AbstractKroneckerMatrix, i::Int, j::Int)

    # Return the entry (i, j) of all d coefficient matrices
    return [ A[s][i, j] for s in 1:length(A) ]

end

function Base.getindex(A::AbstractKroneckerMatrix, multiindex::Matrix{<:Int}) 

    # Implement multiindexing of KroneckerProduct structures. To be precise
    # index a KroneckerProduct structure with a pair of multiindices ℑ₁, ℑ₂.
    d = size(multiindex, 1)

    length(A) == d || throw(BoundsError(A, multiindex))
    
    entries = zeros(d)

    for s = 1:length(A)

        k, l = multiindex[s, :]
        
        entries[s] = A[s][k, l]  

    end

    return entries

end

function Base.setindex!(A::AbstractKroneckerMatrix, M::Matrix, i::Int) 

    1 <= i <= length(A.M) || throw(BoundsError(A, i))

    A.M[i] = M

end

function Base.size(A::AbstractKroneckerMatrix)::Array{Tuple{Int, Int}} 

    # Return size of each KroneckerMatrix element
    
    factor_sizes = Array{Tuple{Int, Int}}(undef, length(A))

    for s = 1:length(A)
        
        factor_sizes[s] = size(A[s])

    end

    return factor_sizes
end

dimensions(A::AbstractKroneckerMatrix) = [ size(A[s], 1) for s in 1:length(A) ]

nentries(A::AbstractKroneckerMatrix) = prod(dimensions(A))

function Base.show(::IO, A::AbstractKroneckerMatrix) 

    d = length(A)
    n = size(A[1], 1)
    println("Kronecker sum of order d = ", d, " with coefficient matrices of order n = ", n)
    println(typeof(A[1]))

end

randkronmat(orders::Array{Int}) = AbstractKroneckerMatrix([ rand(n, n) for n ∈ orders ])

mutable struct KroneckerMatrix{matT, U} <: AbstractKroneckerMatrix{matT}
    
    M          ::CoeffMat{matT} # We only store the d matrices explicitly in a vector.
    matrixclass::Type{<:MatrixGallery}

    function KroneckerMatrix{U}(Aₛ::CoeffMat{matT}) where {matT, U<:Instance}

        new{matT, U}(Aₛ, MatrixGallery)

    end

    function KroneckerMatrix{U}(Aₛ::CoeffMat{matT}, class::Type{<:MatrixGallery}) where {matT, U<:Instance}

        new{matT, U}(Aₛ, class)

    end

    function KroneckerMatrix{U}(Aₛ::CoeffMat) where {U}

        new{eltype(Aₛ), U}(Aₛ, MatrixGallery)

    end

    function KroneckerMatrix{U}(A_view::MatrixView{matT, T}) where {matT, T, U<:Instance}

        new{matT, U}(A_view, MatrixGallery)

    end


    function KroneckerMatrix{matT, U}(orders::Vector{Int}) where {matT, U<:Instance}

        new([ zeros(orders[s], orders[s]) for s in 1:length(orders) ], MatrixGallery)

    end

    function KroneckerMatrix{U}(
        d::Int, n   ::Int,
        class::Type{<:MatrixGallery}) where {U<:Instance}

        A = assemble_matrix(n, class)

        KroneckerMatrix{U}( [ A for _ in 1:d ], class)

    end

    function KroneckerMatrix{U}(
        d::Int, n   ::Int,
        class::Type{MatrixDep}, matrix_id::String) where {U<:Instance}

        A = assemble_matrix(n, class, matrix_id)

        KroneckerMatrix{U}( [ A for _ in 1:d ], class)

    end

    function KroneckerMatrix{U}(d::Int, eigenvalues, class::Type{EigValMat}) where {U<:Instance}

        A = assemble_matrix(eigenvalues, class)
        KroneckerMatrix{U}( [ A for _ in 1:d ], class)

    end

end

mutable struct KroneckerMatrixCompact{matT} <: AbstractKroneckerMatrix{matT}

    M::CoeffMat{matT} # We only store the d matrices explicitly in a vector.

    function KroneckerMatrixCompact{matT}(Aₛ::CoeffMat{matT}) where matT

        new(Aₛ)

    end
    
    function KroneckerMatrixCompact{matT}(A_view::MatrixView{matT, T}) where {matT, T}

        new(A_view)

    end

    function KroneckerMatrixCompact{matT}(orders::Array{Int}) where matT

        new{matT}([ zeros(orders[s], orders[s]) for s in 1:length(orders) ])

    end

end

Base.first(A::KroneckerMatrix{matT,       Instance}) where matT = first(A.M)
Base.first(A::KroneckerMatrixCompact) = first(A.M)
Base.first(A::KroneckerMatrix{matT,    SymInstance}) where matT = Symmetric(first(A.M), :L)
Base.first(A::KroneckerMatrix{matT, NonSymInstance}) where matT = Matrix(first(A.M))

getinstancetype(::KroneckerMatrix{matT, U}) where {matT, U<:Instance} = U

function trikronmat(orders::Array{Int})

    return KroneckerMatrix{Float64, Instance}([ sparse(Tridiagonal( -ones(n - 1), 2ones(n), -ones(n - 1)))  for n in orders ])

end


function kronproddot(v::KronStruct{T}) where T

    return prod( dot(v[s], v[s]) for s in 1:length(v) ) 

end

function kronprodnorm(v::KronStruct{T}) where T

    return sqrt( kronproddot(v) )

end

mutable struct KruskalTensor{T}

    lambda::Vector{T}
    fmat  ::Vector{Matrix{T}}

    function KruskalTensor{T}(lambda::AbstractVector{T}, fmat::KronStruct{T}) where T<:Number

        #@assert all(size(fmat[1], 2) .== size.(fmat, 2))
        #@assert length(lambda) == size(fmat[1], 2)

        new(lambda, fmat)

    end
    
    function KruskalTensor{T}(lambda::Vector{T}, M::Matrix{T}) where T 

        KruskalTensor{T}(lambda, collect(M))

    end

    function KruskalTensor{T}(fmat::KronStruct{T}) where T 

        KruskalTensor{T}(ones(size(fmat[1], 2)), fmat)

    end

    function KruskalTensor{T}() where T

        KruskalTensor{T}(ones(1), [ zeros(1,1) ])

    end


end


ncomponents(x::KruskalTensor{T}) where T = length(x.lambda)

ndims(x::KruskalTensor{T}) where T = length(x.fmat)

Base.size(x::KruskalTensor) = tuple([size(x.fmat[n], 1) for n in 1:length(x.fmat)]...)

function redistribute!(x::KruskalTensor, mode::Int) 

    for j in 1:ncomponents(x)

        x.fmat[mode][:, j] = x.lambda[j] * @view x.fmat[mode][:, j]

    end

end

function display(x::KruskalTensor{T}, name="KruskalTensor") where T

    println("Kruskal tensor of size ", size(x), ":\n")

    println("$name.lambda: ")

        flush(stdout)

        Base.display(x.lambda)

    for n in 1:ndims(x)

        println("\n\n$name.fmat[$n]:")

            flush(stdout)

            Base.display(x.fmat[n])

    end

end

Base.show(::IO, x::KruskalTensor{T}) where T = display(x)



function kroneckervectorize(x::KruskalTensor{T}) where T

    N    = prod(size(x))
    vecx = zeros(N)

    redistribute!(x, 1)

    for i in 1:ncomponents(x) 

        tmp = @view(x.fmat[end][:, i])

        for j in ndims(x) - 1 : - 1 : 1

            tmp = kron(tmp, @view(x.fmat[j][:, i]))

        end

        vecx += tmp

    end

    return vecx

end

function principal_minors(v::KronStruct{T}, i::Int) where T

    return [ @view(v[s][1:i]) for s in 1:length(v) ]

end

function principal_minors(A::KroneckerMatrix{matT, U}, i::Int) where {matT, U}

    return KroneckerMatrix{U}( [ @view(A[s][1:i, 1:i]) for s in 1:length(A)] )

end

function principal_minors(A::KroneckerMatrixCompact{matT}, i::Int, j::Int) where {matT}

    return KroneckerMatrixCompact{matT}( [ @view(A[s][1:i, 1:j]) for s in 1:length(A)] )

end


function principal_minors(x::KruskalTensor{T}, i::Int) where T

    return KruskalTensor{T}(x.lambda, [ (x.fmat[s][1:i, :]) for s in 1:ndims(x) ] )

end


function kth_rows(A::KroneckerMatrix, k::Int)

    return [ @view(A[s][k, :]) for s in 1:length(A) ]

end

function kth_columns(A::KroneckerMatrix, k::Int)

    return [ @view(A[s][:, k]) for s in 1:length(A) ]

end

function kth_columns(A, k::Int)

    return [ @view(A[s][:, k]) for s in 1:length(A) ]

end

# Linear algebra for Kronecker matrices
LinearAlgebra.adjoint(  A::AbstractKroneckerMatrix)   = adjoint.(A.M)
LinearAlgebra.transpose(A::AbstractKroneckerMatrix) = transpose.(A.M)

# Additional functionality for Kruskal tensors
# ============================================
function Base.getindex(CP::KruskalTensor{T}, i::Int) where T

    # Returns a vector containing the i-th column of each factor matrix of CP.

    return [ @view(CP.fmat[s][:, i]) for s = 1:ndims(CP) ]

end

function LinearAlgebra.mul!(result::KronStruct{T}, y::KronStruct{T}, x::KruskalTensor{T}) where T 

    # Compute product between elementary tensor and factor matrices of Kruskal tensor.
    nᵢ = ndims(x)

   for s = 1:nᵢ

       # Result is vector of row vectors
       result[s] = transpose(y[s]) * x.fmat[s]

   end

end

