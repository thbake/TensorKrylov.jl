function mul!(
        result::Vector{Array{T}},
        KP::Vector{Array{T}},
        x::ktensor) where T <: AbstractFloat

    nᵢ = ndims(x)

   for s = 1:nᵢ

       result[s] = transpose(KP[s]) * (x.lambda[s] * x.fmat[s])

   end

end

function mul!(
        result::Vector{Array{T}},
        KP::Vector{Array{T}},
        X::Vector{Matrix{T}}) where T <: AbstractFloat

    nᵢ = length(result)

   for s = 1:nᵢ

       result[s] = transpose(KP[s]) * X.fmat[s]

   end

end

#function solve_compressed_system(
#    H::KroneckerMatrix, 
#    b::AbstractVector, 
#    ω::AbstractArray,
#    α::AbstractArray,
#    t::Int)
#
#    λ = smallest_eigenvalue(H) # This might be different depending on the system
#
#    reciprocal = inv(λ)
#
#    # Since we are considering a canonical decomposition the tensor rank of yₜ
#    # is equal to 
#    yₜ  = TensorStruct{Float64}(undef, (t, dimensions))
#    
#    for j = 1:t
#
#        lhs_coeff = ω[j] * reciprocal
#
#        rhs = Matrix{Float64}(undef, size(H[s])) 
#
#        for s = 1:length(H)
#            
#            rhs_coeff = -α[j] * reciprocal
#            
#            rhs = kron(rhs, exp(coeff .* H[s]) * b[s])
#        end
#
#        yₜ += lhs_coeff * rhs
#    end
#end

function hessenberg_subdiagonals(H::AbstractMatrix, 𝔎::Vector{Int})

    # Extract subdiagonal entries (kₛ₊₁, kₛ) of matrix H⁽ˢ⁾ of ℋ     

    d = length(𝔎)

    entries = Array{Float64}(undef, d)

    for s = 1:d

        entries[s] = H[𝔎[s] + 1, 𝔎[s]]

    end

    return entries

end

function compute_lower_triangle!(
        LowerTriangle::LowerTriangular{T}{Matrix{T}},
        A::Matrix{T},
        B::Matrix{T},
        γ::Array{T},
        k::Int)::Matrix{T} where T <: AbstractFloat

    t = length(γ)

    for j = 1:t-k, i = j+k:t

        LowerTriangle[i, j] = (γ[i]*γ[j])dot(@view(A[:, j]), @view(B[i, :]))
        
    end

end

function compute_lower_triangle!(
        LowerTriangle::LowerTriangular{T}{Matrix{T}},
        A::Matrix{T},
        k::Int)::Matrix{T} where T <: AbstractFloat

    t = size(A, 2)

    for j = 1:t-k, i = j+k:t

        LowerTriangle[i, j] = dot(@view(A[:, j]), @view(A[i, :]))
        
    end

end

function compute_lower_triangle!(
        LowerTriangle::LowerTriangular{T}{Matrix{T}},
        γ::Array{T})::Matrix{T} where T <: AbstractFloat

    t = size(LowerTriangle, 1)

    for j = 1:t, i = j:t

        LowerTriangle[i, j] = γ[i] * γ[j]

    end

end

function compute_coefficients(
        LowerTriangle::LowerTriangular{T}{Matrix{T}},
        δ::Vector{T}) where T <: AbstractFloat

    # Compute Σ |y_𝔏|² with formula in paper, when y is given in CP format:
    #
    # Σ |y_𝔏|² = ||Σᵢ eₖₛᵀ yᵢ⁽ˢ⁾ ⨂ ⱼ≠ ₛ yᵢ⁽ʲ⁾||².
    
    # Get the kₛ-th entry of each column of the s-th factor matrix of y.
    t = length(δ)

    Δ = ones(t, t)

    compute_lower_triangle!(Δ, δ) # ∈ ℝᵗᵗ

    Γ = Δ .* LowerTriangle

    return Γ

end

function matrix_vector(
        H::KroneckerMatrix{T},
        y::ktensor)::AbstractVector where T<:AbstractFloat

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

    for s in eachindex(X)

        X[s] = y.lambda[s] .* X[s] # Scale with lambda

    end

    return X

end


function skipindex(index::Int, range::UnitRange{Int})

    return Iterators.filter(r -> r != index, range)

end

function compressed_residual(
        LowerYY::LowerTriangular{T, Matrix{T}},
        H::KroneckerMatrix{T},
        y::ktensor,
        b::AbstractVector{T}) where T <:AbstractFloat

    # TODO: 
    #
    # We know that 
    #
    # ||Hy - b||² = ||Hy||² -2⋅bᵀ(Hy) + ||b||² 
    
    d = length(H)
    t = ncomponents(y)

    # First we evaluate all X⁽ˢ⁾[:, i] = Hₛy⁽ˢ⁾ᵢ ∈ ℝⁿₛˣᵗ
    X = matrix_vector(H, y)

    LowerXX = repeat([ LowerTriangular(ones(t, t)) ], d)

    compute_lower_triangle!(LowerXX, X, 0)

    XY = repeat([zeros(t,t)], d)

    (mul!(XY[s], transpose(@view(X[s])), y.fmat[s]) for s = 1:d)

    # ||Hy||²

    # Case 1: s = r, i = j:
    # Only compute the squared 2-norms of x⁽ˢ⁾ᵢ for i = 1,…,t

    Hy_norm = sum( tr(LowerXX[s]) for s = 1:eachindex(LowerXX) )

    # Case 2: s = r, i != j:
    # Sum over dot d-1 dot products of the form y⁽ʳ⁾ᵢᵀ y⁽ʳ⁾ⱼ times x⁽ˢ⁾ᵢᵀ x⁽ˢ⁾ⱼ 

    Hy_norm += 2 * sum( LowerYY[r][i, j] * LowerXX[s][i, j] for s = 1:d, r = skipindex(s, 1:d), j = 1:t-1, i = j+1:t )


    # Case 3: s != r, i = j:
    # Only compute two inner products x⁽ʳ⁾ᵢᵀ y⁽ʳ⁾ᵢ times y⁽ˢ⁾ᵢᵀ x⁽ˢ⁾ᵢ
    
    Hy_norm += sum( XY[r][i, i] * XY[s][i, i] for s = 1:d, r = skipindex(s, 1:d), i = 1:t )
    

    # Case 4: s != r, i != j:
    # Compute rest of inner products 

    tmp = 0.0

    for s = 1:d, r = skipindex(s, 1:d)

        for j = 1:t, i = skipindex(j , 1:t)

            tmp += XY[r][i, j] * LowerYY[s][i, j] 

        end

    end

    Hy_norm += 2 * tmp

    # Now we compute <Hy, b>₂

    Hy_b = 0.0

    bY = repeat( [zeros(t)], d )
    bX = repeat( [zeros(t)], d )


    mul!(bY, b, y)
    mul!(bX, b, X)
    


    for s = 1:d, r = skipindex(s, 1:d), i = 1:t

        Hy_b += bY[r][i] * bY[s][i]

    end

    # Finally we compute the 2-norm of b
    b_norm = norm(b)

    return Hy_norm - 2 * Hy_b + b_norm
    
    
end

    
function residual_norm(H::KroneckerMatrix, y::ktensor, 𝔎::Vector{Int}, b)
    
    # Compute norm of the residual according to Lemma 3.4 of paper.
    
    # Σ |hˢₖ₊₁ₖ|² * Σ |y\_𝔏|² + ||ℋy - b̃||²
    
    # Get entries at indices (kₛ+1, kₛ) for each dimension with pair of 
    # multiindices 𝔎+1, 𝔎

    # Number of dimensions
    d = size(H)

    # Tensor rank
    t = ncomponents(y)

    h² = map(abs, hessenberg_subdiagonals(H, 𝔎)).^2

    LowerYY = repeat([ LowerTriangular(ones(t, t)) ], d)

    Λ = LowerTriangular( ones(t, t) )

    compute_lower_triangle!(Λ, y.lambda)


    for s = 1:length(H)

        LowerYY[s] = Λ .* compute_lower_triangle!( LowerYY[s], y.fmat[s], 1)

    end

    res_norm = 0.0

    for s = 1:d

        y² = 0.0

        C = compute_coefficients(LowerYY[s], y.fmat[s][𝔎[s], :])

        for k = 1:t, i = k:t

            product = 1.0

            for j = skipindex(s, 1:d)

                product *= LowerYY[j][i, k]

            end

            y² += C[i, k] * product

        end

        # Here I'm counting the diagonal twice... Need to account for that.
        y² *= 2.0

        res_norm += h²[s] * y²

    end


    # Compute squared compressed residual norm
    rₕ = compressed_residual(LowerYY, H, y, b)

    return res_norm + rₕ

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
