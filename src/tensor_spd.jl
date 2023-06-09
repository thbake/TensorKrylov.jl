export KroneckerMatrix

abstract type KroneckerProduct{T<:AbstractArray} end
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


function dimensions(KP::KroneckerProduct)
    
    factor_dimensions = Array{Int}(undef, length(KP))

    for s = 1:length(KP)
        
        factor_dimensions[s] = size(KP[s], 1)

    end

    return factor_dimensions

end

function nentries(KP::KroneckerProduct)
    
    return prod(dimensions(KP))
    
end
			
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

    function TensorStruct(sizes::Array{Tuple{Int}}, t::Int)

        𝖳ₛ = [ Matrix(undef, shape) for shape in sizes ]

        new{T}(𝖳ₛ, t)
    end

end

struct KroneckerMatrix{T<:AbstractMatrix} <: KroneckerProduct{T} 
    
    𝖳::Vector{T} # We only store the d matrices explicitly in a vector.

    function KroneckerMatrix(Aₛ::Vector{T}) where T<:AbstractMatrix # Constructor with vector of matrix coefficients

        new{T}(Aₛ)

    end

end

function Base.size(KM::KroneckerMatrix)

    # Return size of each KroneckerMatrix element
    
    factor_sizes = Array{Tuple{Int, Int}}(undef, length(KM))

    for s = 1:length(KM)
        
        factor_sizes[s] = size(KM[s])

    end

    return factor_sizes
end

# Additional functionality for Kruskal tensors
# ============================================
function Base.getindex(CP::ktensor, i::Int)

    return [ @view(CP.fmat[s][:, i]) for s = 1:ndims(CP) ]
end

function kron_permutation!(
        b::AbstractVector,
        Xₛ::T,
        Y::AbstractVector{T},
        s::Int,
        dimensions::AbstractVector) where T<:AbstractArray

    # Given an index i compute the following Kronecker product.
    #
    #   CP.fmat[1] ⨂ ⋯ ⨂ CP.fmat[i-1] ⨂ Xᵢ ⨂ CP.fmat[i+1] ⨂ ⋯ ⨂ CP.fmat[d]
    
    d = length(Y) 

    for j = 1:s-2

        kron!( @view(b[1:dimensions[j]]), Y[j], Y[j+1] )

    end

    for j = s+1:d

        kron!( @view(b[1:dimensions[j]]), Xₛ, Y[j])

    end

end

# KroneckerMatrix algebra
# =======================
function mul(A::KroneckerMatrix{T}, x::ktensor)::ktensor where T<:AbstractFloat

    # Allocate memory for resulting matrices B⁽ˢ⁾ resulting from the d matrix-
    # multiplications (or equivalent d * t matrix-vector multiplications) of
    # Aₛ * X⁽ˢ⁾, where X⁽ˢ⁾ are the factor matrices of the CP decomposition of x.
    
    B = TensorStruct(size(A), ndims(x))

    # Allocate memory for large vector of order n_1 ⋯ n_d
    b = Vector{AbstractFloat}(undef, nentries(A))

    n = dimensions(A)

    for s = 1:length(A)

        # Perform multiplication of matrices Aₛ and each factor matrix
        mul!(B[s], A[s], x.fmat[s])

        # Iterate over rank columns
        for i = 1:ncomponents(x)

            # Add resulting vectors over tensor rank
            b += kron_permutation!(b, B[s], x[i], s, n)
        end

    end

    return cp_als(b, ncomponents(x)) # build the result to CP format.
end

function solve_compressed_system(
    H::KroneckerMatrix, 
    b::AbstractVector, 
    ω::AbstractArray,
    α::AbstractArray,
    t::Int)

    λ = smallest_eigenvalue(H) # This might be different depending on the system

    reciprocal = inv(λ)

    # Since we are considering a canonical decomposition the tensor rank of yₜ
    # is equal to 
    yₜ  = TensorStruct{Float64}(undef, (t, dimensions))
    
    for j = 1:t

        lhs_coeff = ω[j] * reciprocal

        rhs = Matrix{Float64}(undef, size(H[s])) 

        for s = 1:length(H)
            
            rhs_coeff = -α[j] * reciprocal
            
            rhs = kron(rhs, exp(coeff .* H[s]) * b[s])
        end

        yₜ += lhs_coeff * rhs
    end
end

function hessenberg_subdiagonals(H::AbstractMatrix, 𝔎::Vector{Int})

    # Extract subdiagonal entries (kₛ₊₁, kₛ) of matrix H⁽ˢ⁾ of ℋ     

    d = length(𝔎)

    entries = Array{Float64}(undef, d)

    for s = 1:d

        entries[s] = H[𝔎[s] + 1, 𝔎[s]]

    end

    return entries

end

function approximate_coefficients(y::ktensor, t::Int, 𝔎::Vector{Int}, s::Int)

    # Approximate Σ |y_𝔏|² with formula in paper, when y is given in CP format.
    
    d = length(y.fmat)

    # Create mask 
    mask = trues(d)

    mask[s] = false

    yʲ = @view y.fmat[mask]


    result = 0

    for i = 1:t

        λᵢ = y.lambda[i]

        α = λᵢ * @view(y.fmat[s][:, i])[𝔎[s]] 

        kronecker_vector = 1

        for j = 1:d-1

            kronecker_vector = kron(kronecker_vector, λᵢ *  @view(yʲ[j][:, i]))

        end

        result += α * kronecker_vector
    end

    # Take the square norm

    result = dot(result, result)
    
end


function compressed_residual(H::KroneckerMatrix, y::ktensor, b::AbstractVector)

    # TODO: Figure out how to perform multiplication between Kronecker matrices
    # and Kruskal Tensors.
    

    
end

    
function residual_norm(H::KroneckerMatrix, y::ktensor, 𝔎::Vector{Int}, b)
    
    # Compute norm of the residual according to Lemma 3.4 of paper.
    
    # Σ |hˢₖ₊₁ₖ|² * Σ |y\_𝔏|² + ||ℋy - b̃||²
    
    # Get entries at indices (kₛ+1, kₛ) for each dimension with pair of 
    # multiindices 𝔎+1, 𝔎

    squared_subdiagonal = map(abs, hessenberg_subdiagonals(H, 𝔎)).^2

    tmp = 0

    for s = 1:length(H)

        squared_y = approximate_coefficients(y, t, 𝔎, s)

        tmp = squared_subdiagonal[s] * squared_y

    end

    compressed_residual(H, y, b)

    
    
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
