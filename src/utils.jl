export TensorizedSystem, solve_tensorized_system
using PyCall

tensor_train = pyimport("scikit_tt.tensor_train")
TT           = pytype_query(tensor_train.TT)
TT_solvers   = pyimport("scikit_tt.solvers.sle")

struct CompressedNormBreakdown{T} <: Exception 
    
    r_comp::T

end

struct TensorizedSystem{T} 

    n                      ::Int
    d                      ::Int
    A                      ::KronMat{T}
    b                      ::KronProd{T}
    orthonormalization_type::Type{<:TensorDecomposition{T}}

    function TensorizedSystem{T}(
        n                     ::Int,
        d                     ::Int,
        orthogonalization_type::Type{<:TensorDecomposition{T}},
        normalize_rhs         ::Bool = true) where T<:AbstractFloat

        Aₛ = assemble_matrix(n, orthogonalization_type)
        bₛ = rand(n)
        A  = KronMat{T}([Aₛ for _ in 1:d])
        b  = [ bₛ for _ in 1:d ]
        
        if normalize_rhs

            normalize!(b)

        end

        new(n, d, A, b, orthogonalization_type)

    end

end


function display(system::TensorizedSystem{T}, name="TensorizedSystem") where T<:AbstractFloat

    println(
        "Tensorized linear system of order d = ",
        system.d,
        "  with coefficient matrices of order n = ",
        system.n
    )
        flush(stdout)

end

function Base.show(io::IO, system::TensorizedSystem{T}) where T<:AbstractFloat

    display(system)

end

function solve_tensorized_system(system::TensorizedSystem{T}, nmax::Int, tol::T = 1e-9) where T<:AbstractFloat

    convergencedata = ConvergenceData{Float64}(nmax)

    tensor_krylov!(
        convergencedata, system.A,
        system.b,
        tol,
        nmax,
        system.orthonormalization_type
    )

    return convergencedata

end

Base.showerror(io::IO, e::CompressedNormBreakdown{T}) where T = print(io, e.r_comp, " is strictly negative.")

function initialize_cores(d::Int, m::Int, n::Int, r1::Int, r2::Int)

    first_core   = zeros(1,  m, n, r2)
    middle_cores = [ zeros(r1, m, n, r2) for _ in 1:d - 2 ]
    final_core   = zeros(r1, m, n, 1)
    cores        = collect( (first_core, middle_cores..., final_core) )

    return cores 

end

function initializeTToperator(Aₛ::AbstractMatrix{T}, d::Int) where T<:AbstractFloat

    n = size(Aₛ, 1)

    cores = initialize_cores(d, n, n, 2, 2)

    cores[1][1, :, :, 1] = Aₛ
    cores[1][1, :, :, 2] = I(n)

    for s in 2:d-1
        
        cores[s][1,:, :, 1] = I(n)
        cores[s][2,:, :, 1] = Aₛ
        cores[s][2,:, :, 2] = I(n)

    end

    cores[end][1, :, :, 1] = I(n)
    cores[end][2, :, :, 1] = Aₛ

    return TT(cores)

end

function initialize_rhs(b::KronProd{T}, d::Int) where T<:AbstractFloat

    
    cores = [ zeros(1, size(b[s], 1), 1, 1) for s in 1:d ]

    for s in 1:d

        cores[s][1, :, 1, 1] = b[s]

    end

    return TT(cores)

end

function canonicaltoTT(x::ktensor)

    d     = ndims(x)
    rank  = ncomponents(x)
    n     = size(x, 1)
    cores = initialize_cores(d, n, 1, rank, rank) 

    tmp = redistribute(x, 1) # Redistribute weights
    #

    for i in 1:rank

        #cores[1][1, :, 1, i] = x.lambda[i] .* @view(x.fmat[1][:, i]) # Fill first core
        #cores[1][1, :, 1, i] = @view(x.fmat[1][:, i]) # Fill first core
        cores[1][1, :, 1, i] = @view(tmp.fmat[1][:, i]) # Fill first core

    end

    for s in 2:d-1, i in 1:rank

        #cores[s][i, :, 1, i] = x.lambda[i] * @view(x.fmat[s][:, i]) # Fill middle cores
        #cores[s][i, :, 1, i] = @view(x.fmat[s][:, i]) # Fill middle cores
        cores[s][i, :, 1, i] = @view(tmp.fmat[s][:, i]) # Fill middle cores

    end

    for i in 1:rank

        #cores[end][i, :, 1, 1] = x.lambda[i] * @view(x.fmat[end][:, i])
        #cores[end][i, :, 1, 1] = @view(x.fmat[end][:, i])
        cores[end][i, :, 1, 1] = @view(tmp.fmat[end][:, i])

    end

    return TT(cores)

end

function TTcompressedresidual(H::KronMat{T}, y::ktensor, b::KronProd{T}) where T<:AbstractFloat

    py"""

    import scikit_tt.tensor_train as tensor_train

    """

    TT = py"tensor_train".TT

    d = length(H)

    H_TT = TT(initializeTToperator(H.𝖳[1], d))
    y_TT = TT(canonicaltoTT(y))
    b_TT = TT(initialize_rhs(b, d))

    TT_multiplication = py"tensor_train.TT.__matmul__"
    TT_subtraction    = py"tensor_train.TT.__sub__"
    TT_norm           = py"tensor_train.TT.norm"

    product    = TT_multiplication(H_TT, y_TT)
    difference = TT_subtraction(product, b_TT)

    return TT_norm(difference)^2
end



function compute_lower_outer!(L::AbstractMatrix{T}, γ::Array{T}) where T <: AbstractFloat

    # Lower triangular matrix representing the outer product of a vector with itself

    t = size(L, 1)

    for j = 1:t, i = j:t

        L[i, j] = γ[i] * γ[j]

    end

end

function cp_tensor_coefficients(Λ::LowerTriangle{T}, δ::Array{T}) where T <: AbstractFloat

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

function skipindex(index::Int, range::UnitRange{Int})

    return Iterators.filter(r -> r != index, range)

end

function maskprod(A::FMatrices{T}, i::Int, j::Int) where T <: AbstractFloat

    # Compute product of entries (i,j) of the matrices contained in A.

    return prod(getindex.(A, i, j)) 

end


function maskprod(x::AbstractVector{<:AbstractVector{T}}, i::Int) where T<:AbstractFloat

    return prod(getindex.(x, i))

end

function maskprod(x::FMatrices{T}, i::Int) where T <: AbstractFloat

    return prod(getindex.(x, 1, i)) 

end

function compute_lower_triangles!(LowerTriangles::FMatrices{T}, x::ktensor) where T<:AbstractFloat

    for s = 1:length(LowerTriangles)

        BLAS.syrk!('L', 'T', 1.0, x.fmat[s], 1.0, LowerTriangles[s])

    end

end

function compute_lower_triangles!(LowerTriangles::FMatrices{T}, x::FMatrices{T}) where T<:AbstractFloat

    for s = 1:length(LowerTriangles)

        BLAS.syrk!('L', 'T', 1.0, x[s], 1.0, LowerTriangles[s])

    end

end

function squared_tensor_entries(Y_masked::FMatrices{T}, Γ::AbstractMatrix{T}) where T <: AbstractFloat

    # Compute Σ |y_𝔏|² with formula in paper, when y is given in CP format:
    #
    #   Σ |y_𝔏|² = ||Σᵢ eₖₛᵀ yᵢ⁽ˢ⁾ ⨂ ⱼ≠ ₛ yᵢ⁽ʲ⁾||², 
    #
    # where δ represents the vector holding kₛ-th entry of each column of the 
    # s-th factor matrix of y.
    #
    # We use the symmetry of the inner products and only require to iterate in
    # the correct way:
    #
    # 2 ⋅Σₖ₌₁ Σᵢ₌ₖ₊₁ Γ[i, k] ⋅ Πⱼ≠ ₛ<yᵢ⁽ʲ⁾,yₖ⁽ʲ⁾> + Σᵢ₌₁ Πⱼ≠ ₛ||yᵢ⁽ʲ⁾||²
    
    t = size(Γ, 1)

    value = 0.0

    for k = 1:t, i = 1:t

        value += Γ[i, k] * maskprod(Y_masked, i, k)

    end

    return value 
end


function matrix_vector(A::KronMat{T}, x::ktensor)::AbstractVector where T<:AbstractFloat

    # Compute the matrix vector products 
    #   
    #   z⁽ˢ⁾ᵢ = Aₛ⋅ x⁽ˢ⁾ᵢ for s = 1,…,d, i = 1,…,t
    #
    # This is equivalent as computing the product Z⁽ˢ⁾ = Aₛ⋅X⁽ˢ⁾, where X⁽ˢ⁾
    # are the factor matrices of the CP-tensor x.

    length(A) == ndims(x) || throw(DimensionMismatch("Kronecker matrix and vector (Kruskal tensor) have different number of components"))

    orders = [ size(A[s], 1) for s in 1:length(A) ]
    rank   = ncomponents(x)

    # Return vector of matrices as described above
    Z = [ zeros(orders[s], rank) for s in eachindex(A) ]

    for s = 1:length(A)

        LinearAlgebra.mul!(Z[s], A[s], x.fmat[s])

    end

    return Z

end

function evalmvnorm(Λ::AbstractMatrix{T}, Y::FMatrices{T}, YZ::FMatrices{T}, Z_inner::FMatrices{T}, i::Int, j::Int, mask_s, mask_r) where T<:AbstractFloat

    return Λ[i, j] * maskprod( Y[.!(mask_s .|| mask_r)], i, j ) *  maskprod(YZ[mask_s .⊻ mask_r], i, j) * maskprod(Z_inner[mask_s .&& mask_r], i, j)

end

function evalinnerprod(y::ktensor, bY::KronProd{T}, bZ::KronProd{T}, i::Int, mask::BitVector) where T<:AbstractFloat

    return y.lambda[i] * maskprod(bZ[mask], i) * maskprod(bY[.!mask], i)

end

function MVnorm(x::ktensor, Λ::AbstractMatrix{T}, lowerX::FMatrices{T}, Z::FMatrices{T}) where T<:AbstractFloat

    Λ_complete = Symmetric(Λ, :L)
    X          = Symmetric.(lowerX, :L)
    Z_inner    = [ Z[s]'Z[s] for s in 1:length(Z) ]
    XZ         = [ x.fmat[s]'Z[s] for s in 1:ndims(x) ]

    d    = length(lowerX)
    rank = length(x.lambda)

    mask_s = falses(d)
    mask_r = falses(d)

    MVnorm = 0.0

    for j = 1:rank, i = 1:rank

        for s in 1:d, r in 1:d

            mask_s[s] = true

            mask_r[r] = true

            MVnorm += evalmvnorm(Λ_complete, X, XZ, Z_inner, i, j, mask_s, mask_r)

            mask_r[r] = false

            mask_s[s] = false

        end

    end

    @assert MVnorm >= 0.0

    return MVnorm
    
end

function tensorinnerprod(Ax::FMatrices{T}, x::ktensor, y::KronProd{T}) where T<:AbstractFloat

    d   = ndims(x)
    t   = ncomponents(x)

    yX  = [ zeros(t) for _ in 1:d ]
    yAx = [ zeros(t) for _ in 1:d ]

    for s in 1:d

        yX[s]  = BLAS.gemv!('T', 1.0, x.fmat[s], y[s], 0.0,  yX[s])
        yAx[s] = BLAS.gemv!('T', 1.0, Ax[s],     y[s], 0.0, yAx[s])

    end

    Ax_y = 0.0

    mask = falses(d)
    
    for s in 1:d

        mask[s] = true

        for i in 1:ncomponents(x)

            Ax_y += evalinnerprod(x, yX, yAx, i, mask)

        end

        mask[s] = false
    end

    @assert Ax_y >= 0.0

    return Ax_y

end

function compressed_residual(H::KronMat{T}, y::ktensor, b::KronProd{T}) where T<:AbstractFloat

    # This variant expands the matrices/tensors

    N = nentries(H)

    H_expanded = sparse(Matrix(kroneckersum(H.𝖳...)))
    y_expanded = reshape(full(y), N)
    b_expanded = kronecker(b...)

    x = zeros(N)

    @assert issparse(H_expanded)

    #mul!(x, H_expanded, y_expanded)

    comp_res = (H_expanded * y_expanded) - b_expanded
    comp_res = x - b_expanded
    
    @info "Compressed residual" dot(comp_res, comp_res)
    return dot(comp_res, comp_res)

end

function compressed_residual(
    Ly              ::FMatrices{T},
    Λ               ::AbstractMatrix{T},
    H               ::KronMat{T},
    y               ::ktensor,
    b               ::KronProd{T}) where T <:AbstractFloat

    # We know that 
    
    #   ||Hy - b||² = ||Hy||² -2⋅bᵀ(Hy) + ||b||² 

    # For this we evaluate all z⁽ˢ⁾ᵢ=  Z⁽ˢ⁾[:, i] = Hₛy⁽ˢ⁾ᵢ ∈ ℝᵏₛ for i = 1,…,t
    Z = matrix_vector(H, y)

    Ly = Symmetric.(Ly, :L)

    # First we compute ||Hy||²
    Hy_norm = MVnorm(y, Symmetric(Λ, :L), Ly, Z)

    # Now we proceed with <Hy, b>₂
    Hy_b = tensorinnerprod(Z, y, b)

    # Finally we compute the squared 2-norm of b
    b_norm = kronproddot(b)

    comp_res = Hy_norm - 2* Hy_b + b_norm

    comp_res < 0.0 ? throw( CompressedNormBreakdown{T}(comp_res) ) : return comp_res

    return comp_res

end


function residual_norm!(
    convergence_data   ::ConvergenceData{T},
    H                  ::KronMat{T},
    y                  ::ktensor,
    𝔎                  ::Vector{Int},
    subdiagonal_entries::Vector{T},
    b                  ::KronProd{T}) where T<:AbstractFloat
    
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

    Ly       = Symmetric.(Ly, :L) # Symmetrize (momentarily abandon the use of symmetry)
    res_norm = 0.0
    mask     = trues(d)

    for s = 1:d

        Γ = Symmetric(cp_tensor_coefficients(Λ, y.fmat[s][𝔎[s], :]), :L) # Symmetric matrix 

        mask[s]   = false
        y²        = squared_tensor_entries(Ly[mask], Γ)
        res_norm += abs( subdiagonal_entries[s] )^2 * y²
        mask[s]   = true

    end

    r_compressed = compressed_residual(Ly, Λ, H, y, b) # Compute squared compressed residual norm
    #r_compressed = TTcompressedresidual(H, y, b)

    return r_compressed, sqrt(res_norm + r_compressed)

end


function normalize!(rhs::KronProd{T}) where T<:AbstractFloat

    for i in 1:length(rhs)

        rhs[i] *= inv(LinearAlgebra.norm(rhs[i]))

    end

end

function initialize_compressed_rhs(b::KronProd{T}, V::KronMat{T}) where T<:AbstractFloat

        b̃        = [ zeros( size(b[s]) )  for s in eachindex(b) ]
        b_minors = principal_minors(b̃, 1)
        columns  = kth_columns(V, 1)
        update_rhs!(b_minors, columns, b, 1)

        return b̃
end

function update_rhs!(b̃::KronProd{T}, V::KronProd{T}, b::KronProd{T}, k::Int) where T<:AbstractFloat
    # b̃ = Vᵀb = ⨂ Vₛᵀ ⋅ ⨂ bₛ = ⨂ Vₛᵀbₛ
    
    for s = 1:length(b̃)

        # Update one entry of each component of b̃ by performing a single inner product 
        b̃[s][k] = dot(V[s], b[s])

    end

end

function basis_tensor_mul!(x::ktensor, V::KronMat{T}, y::ktensor) where T<:AbstractFloat

    x.lambda = copy(y.lambda)

    for s in eachindex(V)

        LinearAlgebra.mul!(x.fmat[s], V[s], y.fmat[s])

    end

end

function compute_minors(tensor_decomp::TensorDecomposition{T}, rhs::KronProd{T}, n::Int, k::Int) where T<:AbstractFloat

        H_minors = principal_minors(tensor_decomp.H, k)
        V_minors = principal_minors(tensor_decomp.V, n, k)
        b_minors = principal_minors(rhs, k)

        return H_minors, V_minors, b_minors
    
end

function matrix_exponential_vector!(y::ktensor, A::KronMat{T}, b::KronProd{T}, γ::T, k::Int) where T<:AbstractFloat

    for s = 1:length(A)

        tmp = Matrix(copy(A[s]))

        #y.fmat[s][:, k] = expv(γ, tmp, b[s]) # Update kth column
        y.fmat[s][:, k] =  exp(γ * tmp) * b[s] # Update kth column

    end

end

function exponentiate(A::AbstractMatrix{T}, γ::T) where T<:AbstractFloat

    tmp    = zeros(size(A))
    result = zeros(size(A))

    λ, V = LinearAlgebra.eigen(A)

    Λ = Diagonal(exp.(γ .* λ))

    LinearAlgebra.mul!(tmp, V, Λ)

    LinearAlgebra.mul!(result, tmp, transpose(V))

    return result

end
