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

function Base.eachindex(KP::KroneckerProduct)

    return eachindex(KP.𝖳)

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

function norm(KP::KroneckerProduct)

    return prod( map(norm, KP) )
end 
			
struct TensorStruct{T<:AbstractArray} <: KroneckerProduct{T}

    𝖳::Vector{T} #\sansT
    rank::Int
    
    function TensorStruct(𝖳ₛ::Vector{T}, t::Int) where T<:AbstractArray
        new{T}(𝖳ₛ, t)
    end

    function TensorStruct(dimensions::Array{Int}, t::Int) where T<:AbstractFloat
        
        # Allocate memory for different arrays in decomposition
        # by giving dimensions of each vector/matrix and tensor rank
        
        𝖳ₛ = [ Array{T}(undef, dimensions[i]) for i = 1:length(dimensions) ]

        new{T}(𝖳ₛ, t)
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

# Linear algebra for KroneckerMatrix
function norm(KM::KroneckerMatrix)

    # This is the best I can think of right now.
    A = kroneckersum(KM)

    return norm(A)

# Additional functionality for Kruskal tensors
# ============================================
function Base.getindex(CP::ktensor, i::Int)

    return [ @view(CP.fmat[s][:, i]) for s = 1:ndims(CP) ]
end


# KroneckerMatrix algebra
# =======================
#function mul(A::KroneckerMatrix{T}, x::ktensor)::ktensor where T<:AbstractFloat
#
#    # Allocate memory for resulting matrices B⁽ˢ⁾ resulting from the d matrix-
#    # multiplications (or equivalent d * t matrix-vector multiplications) of
#    # Aₛ * X⁽ˢ⁾, where X⁽ˢ⁾ are the factor matrices of the CP decomposition of x.
#    
#    B = TensorStruct(size(A), ndims(x))
#
#    # Allocate memory for large vector of order n_1 ⋯ n_d
#    b = Vector{AbstractFloat}(undef, nentries(A))
#
#    n = dimensions(A)
#
#    for s = 1:length(A)
#
#        # Perform multiplication of matrices Aₛ and each factor matrix
#        mul!(B[s], A[s], x.fmat[s])
#
#        # Iterate over rank columns
#        for i = 1:ncomponents(x)
#
#            # Add resulting vectors over tensor rank
#            b += kron_permutation!(b, B[s], x[i], s, n)
#        end
#
#    end
#
#    return cp_als(b, ncomponents(x)) # build the result to CP format.
#end

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

function inner_products(Y::Vector{AbstractMatrix}, s::Int, i::Int, k::Int)

    product = 1

    for j = 1:s-1

        product *=  dot(@view(Y[j][:, i]), @view(Y[j][:, k]))

    end

    for j = s+1:length(Y)

        product *=  dot(@view(Y[j][:, i]), @view(Y[j][:, k]))

    end

    return product

end

function compute_coefficients(y::ktensor, t::Int, 𝔎::Vector{Int}, s::Int)

    # Compute Σ |y_𝔏|² with formula in paper, when y is given in CP format:
    #
    # Σ |y_𝔏|² = ||Σᵢ eₖₛᵀ yᵢ⁽ˢ⁾ ⨂ⱼ\_≠ ₛ yᵢ⁽ʲ⁾||².
    #
    # TODO: Multiply lambdas with the corresponding columns of each factor 
    # matrix.
    
    
    d = length(y.fmat)

    # Create mask 
    mask = trues(d)

    mask[s] = false

    # Create a view of factor matrices that are not indexed by s.
    # This is a Vector of AbstractMatrix.
    Y = @view y.fmat[mask]

    result = 0

    for i = 1:t, k = 1:t

        λᵢ = y.lambda[i]

        λₖ = y.lambda[k]

        αᵢ = λᵢ * @view(y.fmat[s][:, i])[𝔎[s]] 

        αₖ = λₖ * @view(y.fmat[s][:, k])[𝔎[s]] 

        product = inner_products(Y, s, i, k)

        result += (αᵢ * αₖ) * product
        
    end

    return result

end

function matrix_vector(H::KroneckerMatrix{T}, y::ktensor{T})::AbstractVector where T<:AbstractFloat

    # Compute the matrix vector products 
    #   
    #   x⁽ˢ⁾ᵢ = Hₛ⋅ y⁽ˢ⁾ᵢ for s = 1,…,d, i = 1,…,t
    #
    # This is equivalent as computing the product X⁽ˢ⁾ = Hₛ⋅Y⁽ˢ⁾, where Y⁽ˢ⁾
    # are the factor matrices of the CP-tensor y.

    orders = dimensions(H)
    rank   = ncomponents(y)

    # Return vector of matrices as described above
    X = [ AbstractMatrix{T}(undef, (orders[s], rank)) for s in eachindex(H) ]

    for s = 1:length(H)

        mul!(X[s], H[s], y.fmat[s])

    end

    return X

end

function multiple_hadamard!(
        matrices::Vector{Matrix{T}},
        product::Matrix{T}) where T<:AbstractFloat

    for matrix ∈ matrices

        product .*= matrix

    end

end
function compressed_residual(
        H::KroneckerMatrix{T},
        y::ktensor{T},
        b::AbstractVector{T}) where T <:AbstractFloat

    # TODO: 
    #
    # We know that 
    #
    # ||Hy - b||² = ||Hy||² -2⋅bᵀ(Hy) + ||b||² 
    
    # First we evaluate all X⁽ˢ⁾[:, i] = Hₛy⁽ˢ⁾ᵢ ∈ ℝⁿₛˣᵗ
    X = matrix_vector(H, y)

    # ||Hy||²
    d = length(H)
    t = ncomponents(y)

    # Now compute the squared 2-norms of x⁽ˢ⁾ᵢ for i = 1,…,t
    # For this, we need to compute the inner products x⁽ˢ⁾ᵢ of X⁽ˢ⁾, which is 
    # the same as computing X⁽ˢ⁾ᵀ X⁽ˢ⁾, for s = 1,…,d

    Y = @view(y.fmat)

    # Allocate memory for the matrices X⁽ˢ⁾ᵀ X⁽ˢ⁾, Y⁽ˢ⁾ᵀ Y⁽ˢ⁾
    lowerX = repeat( [LowerTriangular( zeros(t, t) )], d )
    lowerY = repeat( [LowerTriangular( ones(t, t)  )], d )
    
    # Allocate memory for the matrices X⁽ˢ⁾ᵀ Y⁽ˢ⁾
    XY = repeat([ zeros(t,t) ], d)


    for s = 1:d

        mul!(XY[s], @view(transpose(X[s])), Y[s])

        # Since X⁽ˢ⁾ᵀ X⁽ˢ⁾ is symmetric, we only compute the lower triangle.
        # Maybe I don't need to perform this step.
        for j = 1:t, i = j:t

            lowerX[s][i, j] = dot(@view( X[s][:, j] ), @view( X[s][i, :] ))


        end

        # Since Y⁽ˢ⁾ᵀ Y⁽ˢ⁾ is symmetric and yᵢᵀyᵢ = 1, we only compute the lower
        # triangle starting at the subdiagonal.
        for j = 1:t-1, i = j+1:t

            lowerY[s][i, j] = dot(@view( Y[s][:, j] ), @view( Y[s][i, :] ))

        end
    end

    # Sum over all traces of X (Σₛ tr( X⁽ˢ⁾ ))
    # Maybe don't need to perform this step either.
    traces_X = sum( map(tr, X) )

    # Allocate memory for products: Want to 
    # (1) Compute the Hadamard product of d² matrices 
    # (2) Take the sum over each matrix
    # (3) Take the sum over all summands


    mask = trues(d)

    # Symmetrize lower triangular part of Y
    SymY = Symmetric(lowerY, :L)

    Hy_norm = 0

    for s = 1:d

        mask[s] = false

        for r = 1:d

            products = ones(t,t) 

            mask[r] = false

            tmp = sum( @view(XY[s]) .* @view(XY[r]) )

            remaining_matrices = @view(SymY[mask])

            # Perform Hadamard product of all matrices not indexed by s.
            multiple_hadamard!(remaining_matrices, products)

            Hy_norm += sum(products) + tmp

            mask[r] = true

        end

        mask[s] = true

    end



    b_norm = norm(b)
    
    
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
