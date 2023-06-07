abstract type KroneckerProduct{T<:AbstractArray} end
# We want to generate an abstract notion of structures that can be represented as Kronecker products
# or sums thereof, since all of these can be represented as vectors of abstract arrays
# or matrices
struct TensorStruct{T<:AbstractArray} <: KroneckerProduct{T}

    𝖳::Vector{T} #\sansT
    rank::Int
    
    function TensorStruct(𝖳ₛ::Vector{T}, t::Int) where T<:AbstractArray
        new{T}(𝖳ₛ, t)
    end

    function TensorStruct(dimensions::Array{Int}, t::Int)
        
        # Allocate memory for different arrays in decomposition
        # by giving dimensions of each vector/matrix and tensor rank
        
        𝖳ₛ = [ Array(undef, dimensions[i]) for i = 1:length(dimensions) ]

        new{𝖳}(𝖳ₛ, t)
    end

end

function Base.length(KP::KroneckerProduct)

    return length(KP.𝖳)

end

function Base.getindex(KP::KroneckerProduct, i::Int)

    1 <= i <= length(KP.𝖳) || throw(BoundsError(KP, i))

    return KP.𝖳[i]

end

function Base.getindex(KP::KroneckerProduct, ℑ::Matrix{Int})

    # Implement multiindexing of KroneckerProduct structures
    #@info "I am being called"
    d = size(ℑ, 1)

    length(KP) == d || throw(BoundsError(KP, ℑ))
    
    entries = zeros(d)

    for i = 1:length(KP)

        k, l = ℑ[i, :]
        
        entries[i] = KP[i][k, l]  

    end

    return entries

end

function Base.size(KP::KroneckerProduct)
    
    factor_sizes = Array{Tuple{Int, Int}}(undef, length(KP))

    #factor_sizes = Array{Int}(undef, length(KP))
    
    for i = 1:length(KP)
        
        factor_sizes[i] = size(KP[i])

    end

    return factor_sizes
end

function dimensions(KP::KroneckerProduct)
    
    factor_dimensions = Array{Int}(undef, length(KP))

    for i = 1:length(KP)
        
        factor_dimensions[i] = size(KP[i], 1)

    end

    return factor_dimensions

end
			
struct KroneckerMatrix{T<:AbstractMatrix} <: KroneckerProduct{T} 
    
    𝖳::Vector{T} # We only store the d matrices explicitly in a vector.

    function KroneckerMatrix(Aₛ::Vector{T}) where T<:AbstractMatrix # Constructor with vector of matrix coefficients
        new{T}(Aₛ)
    end

end

mutable struct Arnoldi{T} # Stores Krylov basis and upper Hessenberg matrix
    const A::Matrix{T}    # Original matrix
    V::Matrix{T}          # Matrix representing basis of Krylov subspace
    H::Matrix{T}          # Upper Hessenberg matrix

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

function tensor_krylov(A::KroneckerMatrix, b::AbstractVector, tol) 

	# Initilialize implicit tensorized Krylov subspace basis and upper Hessenberg 
	# matrix of compressed linear system
	decompositions = [Arnoldi(A[s], size(A[s], 1)) for s = 1:length(A)]

	# Compute basis and Hessenberg factor of each Krylov subspace 𝓚ₖ(Aₛ, bₛ) 
	for s = 1:length(A)
		
		arnoldi_modified!(A[s], b[s], 100, decompositions[s])
		
	end

	return decompositions
end
