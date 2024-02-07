export KroneckerMatrix, KruskalTensor
export ConvDiff, EigValMat, Laplace, LaplaceDense, MatrixGallery, RandSPD 
export Instance, SymInstance, NonSymInstance
export assemble_matrix, dimensions, kroneckervectorize,  kronproddot, 
       kronprodnorm, kth_columns, mul!, ncomponents, ndims, nentries, principal_minors,  
       randkronmat, size, trikronmat 

const KronStruct{T} = Vector{<:AbstractVecOrMat{T}}
const FMatrices{T}  = Vector{<:AbstractMatrix{T}}
#const CoeffMat{T}   = Vector{}

abstract type MatrixGallery{T} end
struct LaplaceDense{T} <: MatrixGallery{T} end
struct Laplace{T}      <: MatrixGallery{T} end 
struct ConvDiff{T}     <: MatrixGallery{T} end
struct EigValMat{T}    <: MatrixGallery{T} end
struct RandSPD{T}      <: MatrixGallery{T} end

abstract type Instance end
struct SymInstance    <: Instance end
struct NonSymInstance <: Instance end

# We want to generate an abstract notion of structures that can be represented as Kronecker products
# or sums thereof, since all of these can be represented as vectors of abstract arrays
# or matrices

struct KroneckerMatrix{T, U} 
    
    𝖳           ::FMatrices{T} # We only store the d matrices explicitly in a vector.
    matrix_class::Type{<:MatrixGallery{T}}

    function KroneckerMatrix{T, U}(Aₛ::FMatrices{T}) where {T, U<:Instance}

        new{T, U}(Aₛ, MatrixGallery{T})

    end

    function KroneckerMatrix{T, U}(
        Aₛ   ::FMatrices{T},
        class::Type{<:MatrixGallery{T}}) where {T, U<:Instance}# Constructor with vector of matrix coefficients

        new{T, U}(Aₛ, class)

    end

    function KroneckerMatrix{T, U}(orders::Array{Int}) where {T, U<:Instance}

        new{T, U}([ zeros(orders[s], orders[s]) for s in 1:length(orders) ], MatrixGallery{T})

    end

    function KroneckerMatrix{T, U}(
        d::Int, n   ::Int,
        matrix_class::Type{<:MatrixGallery{T}}) where {T, U<:Instance}

        A = assemble_matrix(n, matrix_class)

        KroneckerMatrix{T, U}( [ A for _ in 1:d ], matrix_class)

    end

    function KroneckerMatrix{T, U}(d::Int, eigenvalues, class::Type{EigValMat{T}}) where {T, U<:Instance}

        A = assemble_matrix(eigenvalues, class)
        KroneckerMatrix{T, U}( [ A for _ in 1:d ], class)

    end

end
# Basic functions for KroneckerMatrix types
# ==========================================
Base.length(A::KroneckerMatrix)           = length(A.𝖳)

Base.getindex(A::KroneckerMatrix, i::Int) = A.𝖳[i]

Base.eachindex(A::KroneckerMatrix)        = eachindex(A.𝖳)

Base.first(A::KroneckerMatrix{T, Instance}) where T = first(A.𝖳)

Base.first(A::KroneckerMatrix{T, SymInstance}) where T    = Symmetric(first(A.𝖳), :L)
Base.first(A::KroneckerMatrix{T, NonSymInstance}) where T = Matrix(first(A.𝖳))

function Base.getindex(A::KroneckerMatrix{T, U}, multiindex::Matrix{<:Int}) where {T, U<:Instance}

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

function Base.setindex!(A::KroneckerMatrix, M::Matrix, i::Int) 

    1 <= i <= length(A.𝖳) || throw(BoundsError(A, i))

    A.𝖳[i] = M

end

getinstancetype(::KroneckerMatrix{T, U}) where {T, U<:Instance} = U

nentries(A::KroneckerMatrix) = prod(dimensions(A))

function assemble_matrix(n::Int, ::Type{LaplaceDense{T}}) where T

    h  = inv(n + 1)
    Aₛ = inv(h^2) * SymTridiagonal(2ones(n), -ones(n))

    return Aₛ

end

assemble_matrix(n::Int, ::Type{Laplace{T}}) where T = sparse( assemble_matrix(n, LaplaceDense{T}) )


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


    

function Base.show(::IO, A::KroneckerMatrix) 

    d = length(A)
    n = size(A[1], 1)
    println("Kronecker sum of order d = ", d, " with coefficient matrices of order n = ", n)
    println(typeof(A[1]))

end

dimensions(A::KroneckerMatrix) = [ size(A[s], 1) for s in 1:length(A) ]


randkronmat(orders::Array{Int}) = KroneckerMatrix{Float64, Instance}([ rand(n, n) for n ∈ orders ])



function trikronmat(orders::Array{Int})

    return KroneckerMatrix{Float64, Instance}([ sparse(Tridiagonal( -ones(n - 1), 2ones(n), -ones(n - 1)))  for n in orders ])

end

function Base.size(KM::KroneckerMatrix)::Array{Tuple{Int, Int}} 

    # Return size of each KroneckerMatrix element
    
    factor_sizes = Array{Tuple{Int, Int}}(undef, length(KM))

    for s = 1:length(KM)
        
        factor_sizes[s] = size(KM[s])

    end

    return factor_sizes
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
    
    function KruskalTensor{T}(lambda::AbstractVector{T}, M::AbstractMatrix{T}) where T 

        KruskalTensor{T}(lambda, collect(M))

    end

    function KruskalTensor{T}(fmat::KronStruct{T}) where T 

        KruskalTensor{T}(ones(size(fmat[1], 2)), fmat)

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

    return [ @view(v[s][1:i]) for s in 1:length(v)]

end

function principal_minors(KM::KroneckerMatrix{T, U}, i::Int) where {T, U<:Instance}

    return KroneckerMatrix{T, U}( [ @view(KM[s][1:i, 1:i]) for s in 1:length(KM)] )

end

function principal_minors(KM::KroneckerMatrix{T, U}, i::Int, j::Int) where {T, U<:Instance}

    return KroneckerMatrix{T, U}( [ @view(KM[s][1:i, 1:j]) for s in 1:length(KM)] )

end

function principal_minors(x::KruskalTensor{T}, i::Int) where T

    return KruskalTensor{T}(x.lambda, [ (x.fmat[s][1:i, :]) for s in 1:ndims(x) ] )

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

