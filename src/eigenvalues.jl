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

		nrm = norm(diag(Hⱼ, -1))

		if nrm <= tol

			return Hⱼ
		end
		
	end
	
	return Hⱼ
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
