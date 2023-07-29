export KroneckerProduct, KroneckerMatrix, size, kronproddot, kronprodnorm, randkronmat, trikronmat

abstract type KroneckerProduct{T<:AbstractFloat} end
abstract type KroneckerIndex end

struct KroneckerSlice <: KroneckerIndex end
# We want to generate an abstract notion of structures that can be represented as Kronecker products
# or sums thereof, since all of these can be represented as vectors of abstract arrays
# or matrices

# Basic functions for KroneckerProduct types
# ==========================================
function Base.length(KP::KroneckerProduct)

    return length(KP.𝖳)

end

function Base.getindex(KP::KroneckerProduct, i::Int)

    1 <= i <= length(KP.𝖳) || throw(BoundsError(KP, i))

    return KP.𝖳[i]

end

function Base.getindex(KP::KroneckerProduct, IMult::Matrix{Int})

    # Implement multiindexing of KroneckerProduct structures. To be precise
    # index a KroneckerProduct structure with a pair of multiindices ℑ₁, ℑ₂.
    d = size(IMult, 1)

    length(KP) == d || throw(BoundsError(KP, IMult))
    
    entries = zeros(d)

    for s = 1:length(KP)

        k, l = IMult[s, :]
        
        entries[s] = KP[s][k, l]  

    end

    return entries

end

function Base.setindex!(KP::KroneckerProduct, M::Matrix{T}, i::Int) where T<:AbstractFloat

    1 <= i <= length(KP.𝖳) || throw(BoundsError(KP, i))

    KP.𝖳[i] = M

end

function Base.eachindex(KP::KroneckerProduct)

    return eachindex(KP.𝖳)

end


function dimensions(KP::KroneckerProduct)::Array{Int}
    
    factor_dimensions = Array{Int}(undef, length(KP))

    for s = 1:length(KP)
        
        factor_dimensions[s] = size(KP[s], 1)

    end

    return factor_dimensions

end

function nentries(KP::KroneckerProduct)
    
    return prod(dimensions(KP))
    
end

function norm(KP::KroneckerProduct)

    return prod( map(norm, KP) )
end 

function recursivekronecker(A::AbstractMatrix, s::Int, orders::Vector{Int})

    d = length(orders)

    if d == 1

        return A

    elseif s == 1 && d > 1

        return kron(recursivekronecker(A, s, orders[1:d-1]), I(orders[d]))

    else

        return kron(I(orders[1]), recursivekronecker(A, s - 1, orders[2:d]))

    end

end

function recursivekronecker(A::Vector{Matrix{T}}, factor_matrix::Vector{Matrix{T}}, s::Int, i::Int, d::Int) where T<:AbstractFloat

    if d == 1

        return A[s][:, i]

    elseif s == 1 && d > 1

        return kron(recursivekronecker(A[1:d-1], factor_matrix[1:d-1], s, i, d - 1), factor_matrix[d][:, i])

    else

        return kron(factor_matrix[1][:, i], recursivekronecker(A[2:d], factor_matrix[2:d], s - 1, i, d - 1))

    end

end

function explicit_kroneckersum(A::Vector{Matrix{T}}) where T <: AbstractFloat

    orders = [ size(A[s], 1) for s in eachindex(A) ]

    N = prod(orders)

    K = zeros(N, N)

    for s in eachindex(A)

        K += recursivekronecker(A[s], s, orders)

    end

    return K
end
			
struct KroneckerMatrix{T} <: KroneckerProduct{T}
    
    𝖳::Vector{<:AbstractMatrix{T}} # We only store the d matrices explicitly in a vector.

    function KroneckerMatrix{T}(Aₛ::Vector{<:AbstractMatrix{T}}) where T <: AbstractFloat # Constructor with vector of matrix coefficients

        new(Aₛ)

    end

    function KroneckerMatrix{T}(orders::Array{Int}) where T<:AbstractFloat

        new([ zeros(orders[s], orders[s]) for s in 1:length(orders) ])

    end

end

function randkronmat(orders::Array{Int})

        return KroneckerMatrix{Float64}([ rand(order, order) for order in orders ])

end


function trikronmat(orders::Array{Int})

    return KroneckerMatrix{Float64}([ sparse(Tridiagonal( -ones(n - 1), 2ones(n), -ones(n - 1)))  for n in orders ])

end

function Base.size(KM::KroneckerMatrix)::Array{Tuple{Int, Int}}

    # Return size of each KroneckerMatrix element
    
    factor_sizes = Array{Tuple{Int, Int}}(undef, length(KM))

    for s = 1:length(KM)
        
        factor_sizes[s] = size(KM[s])

    end

    return factor_sizes
end

# Linear algebra for KroneckerMatrix
function norm(KM::KroneckerMatrix)

    # This is the best I can think of right now.
    A = kroneckersum(KM)

    return norm(A)

end

#function kronmatttm(KM::KroneckerMatrix, ktensor)::ktensor
#
#
#end

function kronproddot(v::Vector{<:AbstractVector{T}}) where T<:AbstractFloat

    return prod( dot(v[s], v[s]) for s in 1:length(v) ) 

end

function kronprodnorm(v::Vector{Vector{T}}) where T<:AbstractFloat

    return sqrt( kronproddot(v) )

end

function principal_minors(v::Vector{Vector{T}}, i::Int) where T<:AbstractFloat

    return [ @view(v[s][1:i]) for s in 1:length(v)]

end

function principal_minors(KM::KroneckerMatrix{T}, i::Int) where T<:AbstractFloat

    return KroneckerMatrix{T}( [ @view(KM[s][1:i, 1:i]) for s in 1:length(KM)] )

end

function principal_minors(x::ktensor, i::Int) 

    return ktensor(x.lambda, [ (x.fmat[s][1:i, :]) for s in 1:ndims(x) ] )

end


function Base.getindex(KM, j::Int, ::KroneckerSlice)

    return [ @view(KM[s][1:j, 1:j]) for s in 1:length(KM) ]

end


# Additional functionality for Kruskal tensors
# ============================================
function Base.getindex(CP::ktensor, i::Int)

    # Returns a vector containing the i-th column of each factor matrix of CP.

    return [ @view(CP.fmat[s][:, i]) for s = 1:ndims(CP) ]

end

function mul!(result::Vector{<:AbstractMatrix{T}}, y::Vector{<:AbstractVector{T}}, x::ktensor) where T <: AbstractFloat

    # Compute product between elementary tensor and factor matrices of Kruskal tensor.
    nᵢ = ndims(x)

   for s = 1:nᵢ

       # Result is vector of row vectors
       result[s] = transpose(y[s]) * x.fmat[s]

   end

end

function mul!(
        result::Vector{<:AbstractMatrix{T}},
        x::Vector{<:AbstractVector{T}},
        A::Vector{<:AbstractMatrix{T}}) where T <: AbstractFloat

    # Compute product between vector of collection of row vectors and matrices.
    nᵢ = length(result)

    for s = 1:nᵢ

        result[s] = transpose(x[s]) * A[s]

    end

end
