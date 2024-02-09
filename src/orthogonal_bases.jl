export orthonormalize!,  initialize_decomp!, isorthonormal,
       arnoldi_algorithm, lanczos_algorithm, orthogonality_loss
export MGS, TTR

abstract type OrthAlg end
struct MGS <: OrthAlg end # Modified Gram-Schmidt
struct TTR <: OrthAlg end # Three-term recurrence

get_orthogonalization(::TensorArnoldi)       = MGS
get_orthogonalization(::TensorLanczos)       = TTR
get_orthogonalization(::TensorLanczosReorth) = TTR

function orthonormalize!(decomposition::Decomposition, k::Int, ::MGS) 

    n = order(decomposition)
    v = zeros(n)

    mul!(v, decomposition.A, @view decomposition.V[:, k])

    @inbounds for i = 1:k

        decomposition.H[i, k] = dot(v, @view decomposition.V[:, i])
        v                   .-= decomposition.H[i, k] * @view decomposition.V[:, i]
    end

    @inbounds for i = 1:k

        decomposition.H[i, k] += dot(v, decomposition.V[:, i])
        v                     -= dot(v, @view decomposition.V[:, i]) * decomposition.V[:, i]

    end

    decomposition.H[k + 1, k] = norm(v)
    decomposition.V[:, k + 1] = v .* inv(decomposition.H[k + 1, k])
end

function orthonormalize!(lanczos::Lanczos, k::Int, ::TTR) 

    n = order(lanczos)
    u = ones(n)

    # uₖ = Avₖ - βₖ₋₁vₖ₋₁
    mul!(u, lanczos.A, @view lanczos.V[:, k] )
    
    u -= lanczos.β .* lanczos.v

    # γₖ = <uₖ, vₖ>
    lanczos.H[k, k] = dot( u, @view lanczos.V[:, k] )

    # v̂ₖ₊₁ = uₖ - γₖvₖ
    v = u - (lanczos.H[k, k] .* @view lanczos.V[:, k])

    # βₖ = ||v̂ₖ₊₁||
    lanczos.β = norm(v)

    # Update basis vector
    lanczos.β == 0.0 ? lanczos.V[:, k + 1] = zeros(n) : lanczos.V[:, k + 1] = inv(lanczos.β) .* v 

    # Set v to last vector in orthonormalization
    lanczos.v = @view lanczos.V[:, k]

    # Update Jacobi matrix
    update_subdiagonals!(lanczos.H, k, lanczos.β)
    
end

function normalizecolumns!(A::AbstractMatrix{T}) where T<:AbstractFloat

    for column in eachcol(A)

        column ./= norm(column)

    end

    return A

end

function reorthogonalize!(X::MatrixView{T}, lanczos::Lanczos, k::Int) where T<:AbstractFloat

    u = copy(lanczos.V[:, k + 1])
    for i in 1:size(X, 2)

        x  = @view X[:, i]
        h  = dot(u, x)
        u -= h .* x

    end

    lanczos.β = norm(u)
    lanczos.V[:, k + 1] = inv(lanczos.β) *u

end


function orthonormalize!(lanczos::LanczosReorth, k::Int, ::TTR) 
    n = order(lanczos)
    r = ones(n)

    # uₖ = Avₖ - βₖ₋₁vₖ₋₁
    mul!(r, lanczos.A, @view lanczos.V[:, k])
    
    r -= lanczos.β .* lanczos.v

    # γₖ = <uₖ, vₖ>
    lanczos.H[k, k] = dot( r, @view lanczos.V[:, k] )

    # v̂ₖ₊₁ = uₖ - γₖvₖ
    v = r - (lanczos.H[k, k] .* @view lanczos.V[:, k])

    # βₖ = ||v̂ₖ₊₁||
    lanczos.β = norm(v)

    # Update basis vector
    lanczos.β == 0.0 ? lanczos.V[:, k + 1] = zeros(n) : lanczos.V[:, k + 1] = inv(lanczos.β) .* v 

    loss = orthogonality_loss(lanczos.V, k + 1)

    T = eltype(lanczos.V)

    if loss > sqrt(eps(T))

        orthonormalize!(lanczos, k, MGS())

        lanczos.β = lanczos.H[k + 1, k]
        
        lanczos.H[1:k - 2, k] .= 0.0

    end

    # Set v to last vector in orthonormalization
    lanczos.v = @view(lanczos.V[:, k])

    # Update Jacobi matrix
    update_subdiagonals!(lanczos.H, k, lanczos.β)

end


function orthonormalize!(t_decomp::TensorDecomposition, b::KronProd) 
    

    # This method performs the first orthonormalization step of both TensorArnoldi 
    # and TensorLanczos data structures.

    orth_alg = get_orthogonalization(t_decomp)

    @inbounds for s in 1:length(t_decomp)

        # Initialize d Arnoldi/Lanczos decompositions
        decomposition = t_decomp.orthonormalization(t_decomp[s]..., b[s])

        # First orthonormalization step for each of the coefficient matrices
        orthonormalize!(decomposition, 1, orth_alg())

    end

end

function orthonormalize!(t_decomp::TensorDecomposition, k::Int) 

    # This method performs not initial orthonormalization steps for the 
    # TensorArnoldi data structure.

    orth_alg = get_orthogonalization(t_decomp)

    @inbounds for s in 1:length(t_decomp)

        decomposition = t_decomp.orthonormalization(t_decomp[s]..., k)

        orthonormalize!(decomposition, k, orth_alg())

    end

end

function arnoldi_algorithm(A::matT, b::Vector{T}, k::Int) where {matT<:AbstractMatrix, T<:AbstractFloat}

    n = size(A, 1)

    arnoldi = Arnoldi(A, zeros(n, k + 1), zeros(k + 1, k), b)

    for j in 1:k

        orthonormalize!(arnoldi, j, MGS())

    end

    return arnoldi

end


function lanczos_algorithm(A::matT, b::Vector{T}, k::Int) where {matT<:AbstractMatrix, T<:AbstractFloat}

    n = size(A, 1)

    lanczos = Lanczos(A, zeros(n, k), zeros(k, k), b)

    for j in 1:k-1

        orthonormalize!(lanczos, j, TTR())

    end

    return lanczos

end

function lanczos_algorithm(A::matT, b::Vector{T}, k::Int, ::Type{LanczosReorth}) where {matT<:AbstractMatrix, T<:AbstractFloat}

    n = size(A, 1)

    lanczos = LanczosReorth(A, zeros(n, k), zeros(k,k), b)
    
    for j in 1:k-1

        orthonormalize!(lanczos, j, TTR())

    end

    return lanczos

end

function rank_k_update!(result::AbstractMatrix{T}, A::AbstractMatrix{T}, k::Int) where T<:AbstractFloat

    decomposition_basis = @view A[:, 1:k]

    mul!(result, transpose(decomposition_basis), decomposition_basis)

    return Symmetric(result)

end

function rank_k_update(A::KronComp, k::Int) 

    result          = [ zeros(k, k) for _ in 1:length(A) ]
    matrix_products = [ rank_k_update!(result[s], A[s], k) for s in 1:length(A)  ]

    return matrix_products

end

function orthogonality_loss(V::AbstractMatrix{T}, k::Int) where T<:AbstractFloat

    result = zeros(k, k)
    result = rank_k_update!(result, V, k)

    return norm(result - I(k))

end

function isorthonormal(V::AbstractMatrix{T}, k::Int, tol::T = 1e-8)::Bool where T<:AbstractFloat

    loss = orthogonality_loss(V, k)

    return loss < tol # True if loss of orthogonality is less than the given tolerance

end

function isorthonormal(decomposition::Decomposition, k::Int)::Bool 

    return isorthonormal(decomposition.V, k)

end

function isorthonormal(tensor_decomp::TensorDecomposition, k::Int)::Bool 

    boolean = true

    for s in 1:length(tensor_decomp)

        boolean = boolean && isorthonormal(tensor_decomp.V[s], k)

    end

    return boolean
end

