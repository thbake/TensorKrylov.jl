export qr_algorithm, qr_hessenberg, qr_decomposition!, next_coefficients!, 
       sign_changes, initial_interval, bisection, analytic_eigenvalues


export CharacteristicPolynomials

# Data structure d sets of Sturm sequences of polynomials

struct CharacteristicPolynomials{T} 

    coefficients::AbstractArray{<:AbstractArray{<:AbstractArray{T}}}

    function CharacteristicPolynomials{T}(d::Int, first_entries::AbstractArray{T}) where T <: AbstractFloat

        # Initialize vector of coefficients of characteristic polynomials
        coefficients = [ Vector{Vector{T}}(undef, 2) for _ in 1:d ]

        for s in 1:d

            # p₀(λ) = 1, p₁(λ) = γ₁ - λ
            coefficients[s][1] = [1]
            coefficients[s][2] = [first_entries[s], -1.0]

        end

        new(coefficients)

    end

end

function sign_changes(x::T, polynomials::AbstractArray{<:AbstractArray{T}})::Int where T<:AbstractFloat

    sign_change_counter = 0
    current_value       = @evalpoly(x, polynomials[1]...)

    for polynomial in @view(polynomials[2:end])

        evaluation = @evalpoly(x, polynomial...)

        if (evaluation * current_value) < 0.0 # Sign changed

            current_value       = evaluation
            sign_change_counter += 1

        end

    end

    return sign_change_counter

end

function next_coefficients!(characteristic_polynomial::AbstractArray{<:AbstractArray{T}}, j::Int, γ::T, β::T) where T<:AbstractFloat

    # Initialize new coefficients of next characteristic polynomial
    α = ones(j + 1)

    # Access coefficients of polynomial at position j - 1 in the sequence (which
    # is j in Julia).
    p1 = copy(characteristic_polynomial[j])

    α[1]     =  γ * p1[1]
    α[2:j]   =  (γ .* p1[2:j]) - p1[1:j - 1]
    α[j + 1] = -p1[j]

    # Access coefficients of polynomial at position j - 2 in the sequence 
    p2 = characteristic_polynomial[j - 1]
    
    α[1:j-1] -= (β^2 .* p2)

    # Add coefficients of new characteristic polynomial to data structure
    push!(characteristic_polynomial, α)

end
    

function next_polynomial!(γ::AbstractArray{T}, β::AbstractArray{T}, polynomials::CharacteristicPolynomials{T}, j::Int) where T<:AbstractFloat

    # Generate next characteristic polynomial in the Sturm sequence.
    # γ and β are the array of the d diagonal and subdiagonal entries at the 
    # corresponding Jacobi matrices at index j, in the sequence.

    for s in 1:length(polynomials)

        next_coefficients!(polynomials.coefficients[s], γ[s], β[s], j)

    end

end

function initial_interval(γ::AbstractArray{T}, β::AbstractArray{T}) where T<:AbstractFloat

    # Given diagonal and subdiagonal entries, compute initial interval that contains all eigenvalues. 
    # Follows from Gershgorin disks theorem and tridiagonal structure of matrix.

    tmp = Vector(copy(β))
    β1 = abs.( push!(tmp, 0.0) )
    β2 = abs.( pushfirst!(tmp[1:end-1], 0.0) )

    left  = minimum(γ - β1 -  β2)
    right = maximum(γ + β1 +  β2)

    return left, right

end

function bisection(y::T, z::T, n::Int, k::Int, polynomials::AbstractArray{<:AbstractArray{T}}) where T<:AbstractFloat

    # Find the roots of the characteristic polynomial of the (symmetric)
    # tridiagonal matrix T.
    
    # Compute unit roundoff
    u = eps(T) / 2

    x = 0.0

    while abs(z - y) > (u * (abs(y) + abs(z)))

        x = (y + z) / 2

        # Count number of sign changes in the sequence which is equal to the 
        # number of eigenvalues of T that, by the Sturm sequence property are 
        # less than x.
        if sign_changes(x, polynomials) >= (n - k)

            z = x

        else
            y = x

        end

    end

    return x
end

function extreme_tensorized_eigenvalues(A::KronMat{T}, char_poly::CharacteristicPolynomials{T}, k::Int) where T<:AbstractFloat


    λ_min = 0.0
    λ_max = 0.0 

    for s in 1:length(A)

        # Extract diagonal and subdiagonal entries of tridiagonal matrices Aₛ
        pₛ = char_poly.coefficients[s]

        next_coefficients!(pₛ, k, A[s][k, k], A[s][k, k-1])

        y, z = initial_interval(diag(A[s], 0), diag(A[s], 1))

        λ_min += bisection(y, z, k, k, pₛ)
        λ_max += bisection(y, z, k, 1, pₛ)

    end

    return λ_min, λ_max

end

# QR-Iterations

function qr_decomposition!(H::AbstractMatrix, rotations::AbstractVector)

	n = size(H, 1)
	
	for j = 1:(n - 1)

		# Compute Givens rotations to zero out each subdiagonal entry
		Gⱼ = givens(H[j, j], H[j + 1, j], j, j + 1)[1]

		rotations[j] = Gⱼ

		# After applying n-1 Givens rotations H becomes an upper triangular matrix
		H = Gⱼ * H

	end

	for j = 1:n - 1
		
		# Apply the conjugate transpose Givens rotations from the right to
        # upper triangular matrix
		H = H * rotations[j]'
	end

	
		
	return H
end

function qr_hessenberg(A, tol, n_max)
	
	# Keep track of rotations since we want to apply them in reverse order afterwards
	rotations = Vector{LinearAlgebra.Givens}(undef, size(A, 1) - 1)
	
	Hⱼ = copy(A)

	for _ in 1:n_max
		
		# Build QR-decomposition implicitly, with O(n) Givens rotations and compute Aᵢ
		Hⱼ = qr_decomposition!(Hⱼ, rotations)

		nrm = LinearAlgebra.norm(diag(Hⱼ, -1))

		if nrm <= tol

            @info "Convergence of eigenvalues"

            return sort(diag(Hⱼ))
		end
		
	end

    @info "No convergence of eigenvalues"
	
    return sort(diag(Hⱼ))
end

function qr_algorithm(A::AbstractMatrix, tol, n_max)
	
	n = size(A, 1)
	
	Q = I(n)
	
	Â = Q' * A * Q
	
	for k = 1:n_max
		Q, R = qr(Â)
		Â = R * Q

		nrm = LinearAlgebra.norm(diag(Â, -1))
		
		if nrm <= tol

			return Â
		end
	end
	
    return sort(diag(Â))

end

function tensor_qr_algorithm(A::KronMat{T}, tol::T, n_max::Int) where T<:AbstractFloat

    λ_min = 0.0
    λ_max = 0.0

    for s in length(A)

        eigs = qr_algorithm(Matrix(A[s]), tol, n_max)

        λ_min += eigs[1]
        λ_max += eigs[end]

    end

    return λ_min, λ_max

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

function analytic_eigenvalues(A::KronMat{T}) where T<:AbstractFloat

    λ_min = 0.0
    λ_max = 0.0

    for Aₛ in A.𝖳

        n = size(Aₛ, 1)
        h = inv(n + 1)

        λ_min += (2 * inv(h^2)) * (1 - cos(π *     inv(n + 1)))
        λ_max += (2 * inv(h^2)) * (1 - cos(π * n * inv(n + 1)))

    end

    return λ_min, λ_max

end


        
    
