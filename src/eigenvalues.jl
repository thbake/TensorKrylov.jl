export qr_hessenberg

function qr_decomposition!(H::AbstractMatrix, rotations::AbstractVector)

	n = size(H, 1)
	
	for j = 1:(n - 1)

		# Compute Givens rotations to zero out each subdiagonal entry
		Gⱼ = givens(H[j, j], H[j + 1, j], j, j + 1)[1]

		rotations[j] = Gⱼ

		# After applying n-1 Givens rotations H becomes an upper triangular matrix
		H[:] = Gⱼ * H

	end

	for j = 1:n - 1
		
		# Apply the conjugate transpose Givens rotations from the right to
        # upper triangular matrix
		H[:] = H * rotations[j]'
	end

	
		
	return H
end

function qr_hessenberg(A, tol, n_max)
	
	# Keep track of rotations since we want to apply them in reverse order afterwards
	rotations = Vector{LinearAlgebra.Givens}(undef, size(A, 1) - 1)
	
	Hⱼ = copy(A)

	for j = 1:n_max
		
		# Build QR-decomposition implicitly, with O(n) Givens rotations and compute Aᵢ
		Hⱼ = qr_decomposition!(Hⱼ, rotations)

		nrm = LinearAlgebra.norm(diag(Hⱼ, -1))

		if nrm <= tol

            @info "Convergence of eigenvalues"

            return diag(Hⱼ)
		end
		
	end

    @info "No convergence of eigenvalues"
	
    return diag(Hⱼ)
end

function qr_algorithm(A::AbstractMatrix, tol, n_max)
	
	n = size(A, 1)
	
	Q = I(n)
	
	Â = Q' * A * Q
	
	for k = 1:n_max
		Q, R = qr(Â)
		Â = R * Q

		nrm = norm(diag(Â, -1))
		
		if nrm <= tol

			return Â
		end
	end
	
	return Â
end

function hessenberg_eigenvalues(H::AbstractMatrix{T}) where T<:AbstractFloat

    H_tmp = copy(H)
    
	k = size(H, 1)
	
	for j = 1:(k - 1)

		# Compute Givens rotations to zero out each subdiagonal entry
		Gⱼ = givens(H_tmp[j, j], H_tmp[j + 1, j], j, j + 1)[1]

		# After applying n-1 Givens rotations H becomes an upper triangular matrix
		H_tmp = Gⱼ * H_tmp

	end

    # Choose eigenvalues with maximal and minimal magnitude
    eigenvalues = sort( map(abs, diag(H_tmp)) )

    λ_min = eigenvalues[1]
    λ_max = eigenvalues[end]

    return λ_min, λ_max

end

function projected_kronecker_eigenvalues(A::KronMat{T}) where T<:AbstractFloat

    λ_min, λ_max = 0.0, 0.0

    for s in 1:length(A)

        #λ_min_s, λ_max_s = hessenberg_eigenvalues(A[s])

        #λ_min += λ_min_s

        #λ_max += λ_max_s

        #eigenvalues = qr_hessenberg(A[s], 1e-5, 100)
        #eigenvalues = abs.(eigenvalues)
        eigenvalues = abs.(eigvals(Matrix(A[s])))

        λ_min += minimum(eigenvalues)

        λ_max += maximum(eigenvalues)
    end

    return λ_min, λ_max

end

        
    
