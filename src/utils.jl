#using PyCall
#
#tensor_train = pyimport("scikit_tt.tensor_train")
#TT           = pytype_query(tensor_train.TT)
#TT_solvers   = pyimport("scikit_tt.solvers.sle")

struct CompressedNormBreakdown{T} <: Exception 
    
    r_comp::T

end


Base.showerror(io::IO, e::CompressedNormBreakdown{T}) where T = print(io, e.r_comp, " is strictly negative.")

#function initialize_cores(d::Int, m::Int, n::Int, r1::Int, r2::Int)
#
#    first_core   = zeros(1,  m, n, r2)
#    middle_cores = [ zeros(r1, m, n, r2) for _ in 1:d - 2 ]
#    final_core   = zeros(r1, m, n, 1)
#    cores        = collect( (first_core, middle_cores..., final_core) )
#
#    return cores 
#
#end
#
#function initializeTToperator(Aₛ::AbstractMatrix{T}, d::Int) where T
#
#    n = size(Aₛ, 1)
#
#    cores = initialize_cores(d, n, n, 2, 2)
#
#    cores[1][1, :, :, 1] = Aₛ
#    cores[1][1, :, :, 2] = I(n)
#
#    for s in 2:d-1
#        
#        cores[s][1,:, :, 1] = I(n)
#        cores[s][2,:, :, 1] = Aₛ
#        cores[s][2,:, :, 2] = I(n)
#
#    end
#
#    cores[end][1, :, :, 1] = I(n)
#    cores[end][2, :, :, 1] = Aₛ
#
#    return TT(cores)
#
#end
#
#function initialize_rhs(b::KronProd{T}, d::Int) where T
#
#    
#    cores = [ zeros(1, size(b[s], 1), 1, 1) for s in 1:d ]
#
#    for s in 1:d
#
#        cores[s][1, :, 1, 1] = b[s]
#
#    end
#
#    return TT(cores)
#
#end
#
#function canonicaltoTT(x::KruskalTensor{T})
#
#    d     = ndims(x)
#    rank  = ncomponents(x)
#    n     = size(x, 1)
#    cores = initialize_cores(d, n, 1, rank, rank) 
#
#    tmp = redistribute(x, 1) # Redistribute weights
#    
#
#    for i in 1:rank
#
#        #cores[1][1, :, 1, i] = x.lambda[i] .* @view(x.fmat[1][:, i]) # Fill first core
#        #cores[1][1, :, 1, i] = @view(x.fmat[1][:, i]) # Fill first core
#        cores[1][1, :, 1, i] = @view(tmp.fmat[1][:, i]) # Fill first core
#
#    end
#
#    for s in 2:d-1, i in 1:rank
#
#        #cores[s][i, :, 1, i] = x.lambda[i] * @view(x.fmat[s][:, i]) # Fill middle cores
#        #cores[s][i, :, 1, i] = @view(x.fmat[s][:, i]) # Fill middle cores
#        cores[s][i, :, 1, i] = @view(tmp.fmat[s][:, i]) # Fill middle cores
#
#    end
#
#    for i in 1:rank
#
#        #cores[end][i, :, 1, 1] = x.lambda[i] * @view(x.fmat[end][:, i])
#        #cores[end][i, :, 1, 1] = @view(x.fmat[end][:, i])
#        cores[end][i, :, 1, 1] = @view(tmp.fmat[end][:, i])
#
#    end
#
#    return TT(cores)
#
#end
#
#function TTcompressedresidual(H::KronMat{T}, y::KruskalTensor{T}, b::KronProd{T}) where T
#
#    py"""
#
#    import scikit_tt.tensor_train as tensor_train
#
#    """
#
#    TT = py"tensor_train".TT
#
#    d = length(H)
#
#    H_TT = TT(initializeTToperator(H.𝖳[1], d))
#    y_TT = TT(canonicaltoTT(y))
#    b_TT = TT(initialize_rhs(b, d))
#
#    TT_multiplication = py"tensor_train.TT.__matmul__"
#    TT_subtraction    = py"tensor_train.TT.__sub__"
#    TT_norm           = py"tensor_train.TT.norm"
#
#    product    = TT_multiplication(H_TT, y_TT)
#    difference = TT_subtraction(product, b_TT)
#
#    return TT_norm(difference)^2
#end



function compute_lower_outer!(L::AbstractMatrix{T}, γ::Array{T}) where T 

    # Lower triangular matrix representing the outer product of a vector with itself

    t = size(L, 1)

    @inbounds for j = 1:t i = j:t

        L[i, j] = γ[i] * γ[j]

    end

end

function cp_tensor_coefficients(Λ::LowerTriangle{T}, δ::Array{T}) where T 

    # Given a collection of lower triangular matrices containing all values of 
    # λ⁽ˢ⁾corresponding to each factor matrix in the CP-decomposition of the 
    # tensor y, and an array δ containing the k-th entry of a column of said 
    # factor matrices, compute the product of both (see section 3.3. bottom).

    t = length(δ)

    Δ = ones(t, t)

    # Lower triangle of outer product
    compute_lower_outer!(Δ, δ) # ∈ ℝᵗᵗ

    Γ = Δ .* Λ

    return Γ

end

function maskprod(A::KronStruct{T}, i::Int, j::Int) where T

    # Compute product of entries (i,j) of the matrices contained in A.

    return prod(getindex.(A, i, j)) 

end

function maskprod(x::AbstractVector{<:AbstractVector{T}}, i::Int) where T

    return prod(getindex.(x, i))

end

function maskprod(x::FMatrices{T}, i::Int) where T

    return prod(getindex.(x, 1, i)) 

end

function compute_lower_triangles!(LowerTriangles::FMatrices{T}, x::KruskalTensor{T}) where T

    @inbounds for s = 1:length(LowerTriangles)

        BLAS.syrk!('L', 'T', 1.0, x.fmat[s], 1.0, LowerTriangles[s])

    end

end

function compute_lower_triangles!(LowerTriangles::FMatrices{T}, M::FMatrices{T}) where T

    @inbounds for s = 1:length(LowerTriangles)

        BLAS.syrk!('L', 'T', 1.0, M[s], 1.0, LowerTriangles[s])

    end

end

function squared_tensor_entries(Y_masked::FMatrices{T}, Γ::AbstractMatrix{T}) where T

    t = size(Γ, 1)

    value = 0.0

    @inbounds for k = 1:t

        value += Γ[k, k] .* maskprod(Y_masked, k, k)

        for i = k+1:t

            value +=  2 .* Γ[i, k] .* maskprod(Y_masked, i, k)

        end

    end

    return value

end


function matrix_vector(A::KronMat, x::KruskalTensor) 

    # Compute the matrix vector products 
    #   
    #   z⁽ˢ⁾ᵢ = Aₛ⋅ x⁽ˢ⁾ᵢ for s = 1,…,d, i = 1,…,t
    #
    # This is equivalent as computing the product Z⁽ˢ⁾ = Aₛ⋅X⁽ˢ⁾, where X⁽ˢ⁾
    # are the factor matrices of the CP-tensor x.

    length(A) == ndims(x) || throw(DimensionMismatch("Kronecker matrix and vector (Kruskal tensor) have different number of components"))

    orders = [size(A[s], 1) for s in 1:length(A) ]
    rank   = ncomponents(x)

    Z = [ zeros(orders[s], rank) for s in eachindex(A) ]

    @inbounds for s = 1:length(A)

        LinearAlgebra.mul!(Z[s], A[s], x.fmat[s])

    end

    return Z

end

function f(X, mask_s::BitVector, mask_r::BitVector, i::Int, j::Int) 

    mask = mask_s .⊻ mask_r
    
    return !any(mask) ? 1.0 : maskprod(X[mask_s], i, j) * maskprod(X[mask_r], j, i)

end

function evalmvnorm(
    Λ ::AbstractMatrix{T},
    Ly::FMatrices{T},
    X ::FMatrices{T},
    Lz::FMatrices{T},
    i::Int, j::Int, 
    mask_s, mask_r)::T where T

    α = maskprod(Ly[.!(mask_s .|| mask_r)], i, j)
    β = f(X, mask_s, mask_r, i, j)
    γ = maskprod(Lz[mask_s .&& mask_r], i, j)

    return Λ[i, j] * α * β * γ

end


function MVnorm(y::KruskalTensor, Λ::AbstractMatrix, Ly, Z) 

    d    = length(Ly)
    rank = length(y.lambda)

    Lz         = [ zeros(rank, rank) for _ in 1:d ]
    X          = [ y.fmat[s]'Z[s]    for s in 1:d ]

    compute_lower_triangles!(Lz, Z) # Compute lower triangular parts
    

    mask_s = falses(d)
    mask_r = falses(d)

    MVnorm = 0.0

    @inbounds for s in 1:d, r in 1:d

        mask_s[s] = true

        mask_r[r] = true

        @inbounds for j = 1:rank

            MVnorm += @inline evalmvnorm(Λ, Ly, X, Lz, j, j, mask_s, mask_r)

            for i = j+1:rank

                MVnorm += @inline 2 * evalmvnorm(Λ, Ly, X, Lz, i, j, mask_s, mask_r)

            end

        end

        mask_r[r] = false

        mask_s[s] = false

    end

    #@assert MVnorm >= 0.0

    return MVnorm
    
end

function evalinnerprod(y::KruskalTensor, bY::KronProd{T}, bZ::KronProd{T}, i::Int, mask::BitVector) where T

    return y.lambda[i] * maskprod(bZ[mask], i) * maskprod(bY[.!mask], i)

end

function tensorinnerprod(Z::FMatrices{T}, y::KruskalTensor{T}, b_norm::T) where T

    d   = ndims(y)
    t   = ncomponents(y)

    bY = [ zeros(t) for _ in 1:d ]
    bZ = [ zeros(t) for _ in 1:d ]

    for s in 1:d

        bY[s] = y.fmat[s][1, :] # First row vector contains first entries
        bZ[s] = Z[s][1, :]

    end

    Ax_b = 0.0

    mask = falses(d)
    
    @inbounds for s in 1:d

        mask[s] = true

        for i in 1:t

            Ax_b += @inline evalinnerprod(y, bY, bZ, i, mask)

        end

        mask[s] = false
    end

    #@assert Ax_y >= 0.0
    Ax_b = Ax_b * b_norm

    return Ax_b

end

function compressed_residual(
    Ly::FMatrices,
    Λ ::AbstractMatrix,
    H ::KronMat,
    y ::KruskalTensor,
    b̃,
    b_norm)

    # We know that 
    
    #   ||Hy - b||² = ||Hy||² -2⋅bᵀ(Hy) + ||b||² 

    # For this we evaluate all z⁽ˢ⁾ᵢ=  Z⁽ˢ⁾[:, i] = Hₛy⁽ˢ⁾ᵢ ∈ ℝᵏₛ for i = 1,…,t

    # Return vector of matrices as described above

    Z = matrix_vector(H, y)

    Hy_norm = MVnorm(y, Λ, Ly, Z) # First we compute ||Hy||²
    #Hy_b    = tensorinnerprod(Z, y, b)           # <Hy, b>₂
    Hy_b    = tensorinnerprod(Z, y, b_norm)           # <Hy, b>₂
    b_norm  = kronproddot(b̃)                     # squared 2-norm of b
    r_comp  = Hy_norm - 2* Hy_b + b_norm

    r_comp < 0.0 ? throw( CompressedNormBreakdown{eltype(Λ)}(r_comp) ) : return r_comp

    return r_comp

end


function residualnorm!(
    H,                  
    y                  ::KruskalTensor,
    𝔎                  ::Vector{Int},
    subdiagonal_entries::Vector,
    b̃,
    b_norm) 
    
    # Compute squared norm of the residual according to Lemma 3.4 of paper.
    
    # Σ |hˢₖ₊₁ₖ|² * Σ |y\_𝔏|² + ||ℋy - b̃||²
    
    # Get entries at indices (kₛ+1, kₛ) for each dimension with pair of 
    # multiindices 𝔎+1, 𝔎

    d  = length(H)                    # Number of dimensions
    t  = ncomponents(y)               # Tensor rank
    Ly = [ zeros(t, t) for _ in 1:d ] # Allocate memory for matrices representing inner products
    Λ  = LowerTriangular(zeros(t, t)) # Allocate memory for matrix representing outer product of coefficients.

    compute_lower_triangles!(Ly, y)
    compute_lower_outer!(Λ, y.lambda)

    res_norm = 0.0
    mask     = trues(d)

    @inbounds for s = 1:d

        Γ = cp_tensor_coefficients(Λ, y.fmat[s][𝔎[s], :])

        mask[s]   = false
        y²        = squared_tensor_entries(Ly[mask], Γ)
        res_norm += abs2( subdiagonal_entries[s] ) * y²
        mask[s]   = true

    end

    r_comp = compressed_residual(Ly, Λ, H, y, b̃, b_norm) # Compute squared compressed residual norm

    return r_comp, sqrt(res_norm + r_comp)

end


function LinearAlgebra.normalize!(rhs::KronProd{T}) where T

    for i in 1:length(rhs)

        rhs[i] *= inv(LinearAlgebra.norm(rhs[i]))

    end

end

function initialize_compressed_rhs(b::KronProd, V::KronComp) 

        b̃        = [ zeros( size(b[s]) )  for s in eachindex(b) ]
        b_minors = principal_minors(b̃, 1)
        columns  = kth_columns(V, 1)
        update_rhs!(b_minors, columns, b, 1)

        return b̃
end

function update_rhs!(b̃, V, b, k::Int) 
    # b̃ = Vᵀb = ⨂ Vₛᵀ ⋅ ⨂ bₛ = ⨂ Vₛᵀbₛ
    
    @inbounds for s in eachindex(b̃)

        # Update one entry of each component of b̃ by performing a single inner product 
        b̃[s][k] = dot(V[s], b[s])

    end

end

function basis_tensor_mul!(x::KruskalTensor{T}, V::KronComp, y::KruskalTensor{T})  where T

    x.lambda = copy(y.lambda)

    @inbounds for s in eachindex(V)

        LinearAlgebra.mul!(x.fmat[s], V[s], y.fmat[s])

    end

end

function compute_minors(tensor_decomp::TensorDecomposition, rhs::KronProd, n::Int, k::Int) 

        H_minors = principal_minors(tensor_decomp.H, k)
        V_minors = principal_minors(tensor_decomp.V, n, k)
        b_minors = principal_minors(rhs, k)

        return H_minors, V_minors, b_minors
    
end


function matrix_exponential_vector!(
    y::KruskalTensor{T},
    A,
    b,
    γ::T, 
    k::Int,
    ::Type{<:MatrixGallery}) where T

    tmp = γ * first(A)

    expA = exp(tmp)

    @sync for s = 1:length(A)

        @async begin

            @inbounds y.fmat[s][:, k] =  expA * b[s] # Update kth column

        end

    end

end

function matrix_exponential_vector!(
    y::KruskalTensor{T},
    A,
    b,
    γ::T, 
    k::Int,
    ::Type{EigValMat}) where T


    @sync for s in 1:length(A)

        @async begin

            D = exp(γ .* A[s])

            @inbounds y.fmat[s][:, k] =  D * b[s] # Update kth column

        end

    end

end

exponentiate_diagonal(D) = Diagonal(exp.(diag(D)))

