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

function scale_matrices!(Aₛ::Vector{Matrix{T}}, Λ::Matrix{T}) where T <: AbstractFloat

    for s in eachindex(Aₛ)

        Aₛ[s] .*= Λ

    end

end

function compute_lower_triangle!(
        LowerTriangle::LowerTriangular{T, Matrix{T}},
        A::Matrix{T},
        B::Matrix{T},
        γ::Array{T},
        k::Int) where T <: AbstractFloat

    t = length(γ)

    for j = 1:t-k, i = j+k:t

        LowerTriangle[i, j] = (γ[i]*γ[j])dot(@view(A[:, j]), @view(B[:, i]))
        
    end

end

function compute_lower_triangle!(
        LowerTriangle::LowerTriangular{T, Matrix{T}},
        A::Matrix{T},
        k::Int) where T <: AbstractFloat

    t = size(A, 2)

    for j = 1:t-k, i = j+k:t

        LowerTriangle[i, j] = dot(@view(A[:, j]), @view(A[:, i]))
        
    end

end

function compute_lower_triangle!(
        LowerTriangle::LowerTriangular{T, Matrix{T}},
        γ::Array{T}) where T <: AbstractFloat

    t = size(LowerTriangle, 1)

    for j = 1:t, i = j:t

        LowerTriangle[i, j] = γ[i] * γ[j]

    end

end

function compute_coefficients(
        LowerTriangle::LowerTriangular{T, Matrix{T}},
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

function innerproducts!(
        LowerTriangles::Vector{LowerTriangular{T, Matrix{T}}},
        factormatrix,
        k::Int) where T <: AbstractFloat

    for s in eachindex(LowerTriangles)

        compute_lower_triangle!(LowerTriangles[s], factormatrix[s], k)

    end
    
end

function matrix_vector(
        A::KroneckerMatrix{T},
        x::ktensor)::AbstractVector where T<:AbstractFloat

    # Compute the matrix vector products 
    #   
    #   z⁽ˢ⁾ᵢ = Aₛ⋅ x⁽ˢ⁾ᵢ for s = 1,…,d, i = 1,…,t
    #
    # This is equivalent as computing the product Z⁽ˢ⁾ = Aₛ⋅X⁽ˢ⁾, where X⁽ˢ⁾
    # are the factor matrices of the CP-tensor x.

    orders = dimensions(A)
    rank   = ncomponents(x)

    # Return vector of matrices as described above
    Z = [ zeros(orders[s], rank) for s in eachindex(A) ]

    for s = 1:length(A)

        LinearAlgebra.mul!(Z[s], A[s], x.fmat[s])

    end

    for s = 1:length(A)

        # Scale columns of each matrix with entries of lambda
        Z[s] = x.lambda' .* Z[s] 

    end

    return Z

end

function skipindex(index::Int, range::UnitRange{Int})

    return Iterators.filter(r -> r != index, range)

end

function ktensor_innerprods!(
        Lx::Vector{LowerTriangular{T, Matrix{T}}}, 
        x::ktensor) where T <: AbstractFloat

    t = ncomponents(x)

    # Allocate memory for and initialize matrix representing scaling of factor 
    # matrices in the CP decomposition
    Λ = LowerTriangular( ones(t, t) )

    compute_lower_triangle!(Λ, x.lambda)

    # Compute (lower triangular) matrices representing inner products
    innerproducts!(Lx, x.fmat, 1)

    # Scale with norms of CP decomposition
    map(X -> Λ .* X, Lx)

end

function squared_matrix_vector(
        Lx::Vector{LowerTriangular{T, Matrix{T}}},
        Z::Vector{Matrix{T}},
        A::KroneckerMatrix{T}, 
        x::ktensor)::T where T <: AbstractFloat

    # Computes the squared norm ||Ax||², where Ax = z

    d = length(A)
    t = ncomponents(x)

    Lz = [ LowerTriangular(ones(t, t)) for _ in 1:d ]

    # Compute (lower triangular) matrix represeting inner products z⁽ˢ⁾ᵢᵀz⁽ˢ⁾ⱼ
    innerproducts!(Lz, Z, 0)

    # Case 1: s = r, i = j:
    # Only sum over the squared 2-norms of z⁽ˢ⁾ᵢ for i = 1,…,t
    squared_norm = sum( tr(Lz[s]) for s in eachindex(Lz) )

    # Case 2: s = r, i != j:
    # Sum over dot d-1 dot products of the form x⁽ʳ⁾ᵢᵀ x⁽ʳ⁾ⱼ times z⁽ˢ⁾ᵢᵀ z⁽ˢ⁾ⱼ 
    for s = 1:d, r = skipindex(s, 1:d)

        for j = 1:t-1, i = j+1:t

            squared_norm += Lx[r][i, j] * Lz[s][i, j]

        end

    end

    # Here we count twice because of symmetry of the inner products
    squared_norm *= 2

    # Case 3: s != r, i = j:
    # Only compute two inner products z⁽ʳ⁾ᵢᵀ x⁽ʳ⁾ᵢ times x⁽ˢ⁾ᵢᵀ z⁽ˢ⁾ᵢ and sum
    # over them


    XZ = [ zeros(t,t) for _ in 1:d ]

    (mul!(XZ[s], transpose(@view(Z[s])), x.fmat[s]) for s = 1:d)

    for s = 1:d, r = skipindex(s, 1:d) 

        for i = 1:t

            squared_norm += XZ[r][i, i] * XZ[s][i, i]

        end

    end

    # Case 4: s != r, i != j:
    # Compute rest of inner products 

    tmp = 0.0

    for s = 1:d, r = skipindex(s, 1:d)

        for j = 1:t, i = skipindex(j , 1:t)

            tmp += XZ[r][i, j] * Lx[s][i, j] 

        end

    end

    squared_norm += 2 * tmp

    return squared_norm

end

function compressed_residual(
        Ly::Vector{LowerTriangular{T, Matrix{T}}},
        H::KroneckerMatrix{T},
        y::ktensor,
        b::Vector{Array{T}}) where T <:AbstractFloat

    # TODO: 
    #
    # We know that 
    #
    # ||Hy - b||² = ||Hy||² -2⋅bᵀ(Hy) + ||b||² 
    
    d = length(H)
    t = ncomponents(y)

    # For this we evaluate all B⁽ˢ⁾[:, i] = Hₛy⁽ˢ⁾ᵢ ∈ ℝⁿₛˣᵗ
    B = matrix_vector(H, y)

    # First we compute ||Hy||²
    Hy_norm = squared_matrix_vector(Ly, B, H, y)

    # Now we compute <Hy, b>₂

    Hy_b = 0.0

    bY = repeat( [zeros(t)], d )
    bZ = repeat( [zeros(t)], d )


    mul!(bY, b, y)
    mul!(bZ, b, B)
    


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

    # Extract subdiagonal entries of upper Hesseberg matrices
    h² = map(abs, hessenberg_subdiagonals(H, 𝔎)).^2

    # Allocate memory for (lower triangular) matrices representing inner products
    Ly = [ LowerTriangular(ones(t, t)) for _ in 1:d ]

    ktensor_innerprods!(Ly, y)

    res_norm = 0.0

    for s = 1:d

        y² = 0.0

        C = compute_coefficients(Ly[s], y.fmat[s][𝔎[s], :])

        for k = 1:t, i = k:t

            product = 1.0

            for j = skipindex(s, 1:d)

                product *= Ly[j][i, k]

            end

            y² += C[i, k] * product

        end

        # Here I'm counting the diagonal twice... Need to account for that.
        y² *= 2.0

        res_norm += h²[s] * y²

    end


    # Compute squared compressed residual norm
    rₕ = compressed_residual(Ly, H, y, b)

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
