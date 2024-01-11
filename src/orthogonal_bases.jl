export orthonormal_basis!, orthonormal_basis_vector!, 
       initial_orthonormalization!, initialize_decomp!, isorthonormal,
       arnoldi_algorithm, lanczos_algorithm, orthogonality_loss

function orthonormal_basis_vector!(decomposition::Decomposition{T}, k::Int, ::MGS) where T<:AbstractFloat

    n = order(decomposition)
    v = zeros(n)

    mul!(v, decomposition.A, @view decomposition.V[:, k])

    for i = 1:k

        decomposition.H[i, k] = dot(v, @view decomposition.V[:, i])
        v                   .-= decomposition.H[i, k] * @view decomposition.V[:, i]
    end

    for i = 1:k

        decomposition.H[i, k] += dot(v, decomposition.V[:, i])
        v                     -= dot(v, @view decomposition.V[:, i]) * decomposition.V[:, i]

    end

    decomposition.H[k + 1, k] = norm(v)
    decomposition.V[:, k + 1] = v .* inv(decomposition.H[k + 1, k])
end

function orthonormal_basis_vector!(lanczos::Lanczos{T}, k::Int) where T<:AbstractFloat

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

#function goodRitzVectors(
#    reorth_struct::Reorthogonalize{T},
#    lanczos      ::Lanczos{T}, β::T, k::Int) where T<:AbstractFloat
#
#    n = size(lanczos.V, 1)
#    Θ, Z = eigen(@view lanczos.H[1:k, 1:k]) # Compute eigenpairs pairs {θᵢ, zᵢ}
#
#    good_threshold    = abs(β) .* abs.(@view Z[k, :])
#    machine_threshold = sqrt(reorth_struct.u) * norm(lanczos.A)
#    println("Machine threshold ", machine_threshold)
#    indices           = good_threshold .<= machine_threshold
#    println(indices)
#    m                 = sum(indices)
#    Y    = zeros(n, k)
#    mul!(Y, @view(lanczos.V[:, 1:k]), Z)
#
#    goodRitz = normalizecolumns!(Y[:, indices])
#
#    return goodRitz
#
#end

function reorthogonalize!(X::MatrixView{T}, lanczos::Lanczos{T}, k::Int) where T<:AbstractFloat

    u = copy(lanczos.V[:, k + 1])
    for i in 1:size(X, 2)

        x  = @view X[:, i]
        h  = dot(u, x)
        u -= h .* x

    end

    lanczos.β = norm(u)
    lanczos.V[:, k + 1] = inv(lanczos.β) *u

end


function orthonormal_basis_vector!(
    lanczos      ::LanczosReorth{T},
    k            ::Int) where T<:AbstractFloat
    
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

    if loss > sqrt(eps(T))

        orthonormal_basis_vector!(lanczos, k, MGS())

        lanczos.β = lanczos.H[k + 1, k]
        
        lanczos.H[1:k - 2, k] .= 0.0

    end

    # Set v to last vector in orthonormalization
    lanczos.v = @view(lanczos.V[:, k])

    # Update Jacobi matrix
    update_subdiagonals!(lanczos.H, k, lanczos.β)

end


function initial_orthonormalization!(
    t_decomp   ::TensorDecomposition{T},
    b          ::KronProd{T},
    decomp_type::Type{<:Decomposition{T}}) where T<:AbstractFloat
    

    # This method performs the first orthonormalization step of both TensorArnoldi 
    # and TensorLanczos data structures.

    for s in 1:length(t_decomp)

        # Initialize d Arnoldi/Lanczos decompositions
        decomposition = decomp_type(t_decomp[s]..., b[s])

        # First orthonormalization step for each of the coefficient matrices
        if decomp_type == Arnoldi{T} 

            orthonormal_basis_vector!(decomposition, 1, MGS())

        else

            orthonormal_basis_vector!(decomposition, 1)
        end

    end

end

function orthonormal_basis!(t_arnoldi::TensorArnoldi{T}, k::Int) where T<:AbstractFloat

    # This method performs not initial orthonormalization steps for the 
    # TensorArnoldi data structure.

    for s in 1:length(t_arnoldi)

        arnoldi = Arnoldi{T}(t_arnoldi[s]...)

        orthonormal_basis_vector!(arnoldi, k, MGS())

    end

end

function orthonormal_basis!(t_lanczos::TensorLanczos{T}, k::Int) where T<:AbstractFloat

    # This method performs not initial orthonormalization steps for the 
    # TensorLanczos data structure.

    for s in 1:length(t_lanczos)

        lanczos = Lanczos{T}(t_lanczos[s]..., k)

        orthonormal_basis_vector!(lanczos, k)

    end

end

function orthonormal_basis!(t_lanczos::TensorLanczosReorth{T}, k::Int) where T<:AbstractFloat

    # This method performs not initial orthonormalization steps for the 
    # TensorLanczos data structure.

    for s in 1:length(t_lanczos)

        lanczos = LanczosReorth{T}(t_lanczos[s]..., k)

        orthonormal_basis_vector!(lanczos, k)

    end

end

function arnoldi_algorithm(A::AbstractMatrix{T}, b::AbstractVector{T}, k::Int) where T<:AbstractFloat

    n = size(A, 1)

    arnoldi = Arnoldi{T}(A, zeros(n, k + 1), zeros(k + 1, k), b)

    for j in 1:k

        orthonormal_basis_vector!(arnoldi, j, MGS())

    end

    return arnoldi

end


function lanczos_algorithm(A::AbstractMatrix{T}, b::AbstractVector{T}, k::Int) where T<:AbstractFloat
    n = size(A, 1)

    lanczos = Lanczos{T}(A, zeros(n, k), zeros(k, k), b)

    for j in 1:k-1

        orthonormal_basis_vector!(lanczos, j)

    end

    return lanczos

end

function lanczos_algorithm(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    k::Int,
     ::MGS) where T<:AbstractFloat

    n = size(A, 1)

    lanczos = LanczosReorth{T}(A, zeros(n, k), zeros(k, k), b)
    
    for j in 1:k-1

        orthonormal_basis_vector!(lanczos, j)

    end

    return lanczos

end

function rank_k_update!(result::AbstractMatrix{T}, A::AbstractMatrix{T}, k::Int) where T<:AbstractFloat

    decomposition_basis = @view A[:, 1:k]

    mul!(result, transpose(decomposition_basis), decomposition_basis)

    return Symmetric(result)

end

function rank_k_update(A::KronMat{T, U}, k::Int) where {T<:AbstractFloat, U<:Instance}

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

function isorthonormal(decomposition::Decomposition{T}, k::Int)::Bool where T<:AbstractFloat

    return isorthonormal(decomposition.V, k)

end

function isorthonormal(tensor_decomp::TensorDecomposition{T}, k::Int)::Bool where T<:AbstractFloat

    boolean = true

    for s in 1:length(tensor_decomp)

        boolean = boolean && isorthonormal(tensor_decomp.V[s], k)

    end

    return boolean
end

