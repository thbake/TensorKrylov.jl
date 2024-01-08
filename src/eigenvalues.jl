using Combinatorics
export analytic_eigenvalues, assemble_matrix,bisection, initial_interval,
       kronsumeigs, next_coefficients!, sign_changes,  qr_algorithm


# Data structure d sets of Sturm sequences of polynomials

#struct CharacteristicPolynomials{T} 
#
#    coefficients::AbstractArray{<:AbstractArray{<:AbstractArray{T}}}
#
#    function CharacteristicPolynomials{T}(d::Int, first_entries::AbstractArray{T}) where T <: AbstractFloat
#
#        # Initialize vector of coefficients of characteristic polynomials
#        coefficients = [ Vector{Vector{T}}(undef, 2) for _ in 1:d ]
#
#        for s in 1:d
#
#            # p₀(λ) = 1, p₁(λ) = γ₁ - λ
#            coefficients[s][1] = [1]
#            coefficients[s][2] = [first_entries[s], -1.0]
#
#        end
#
#        new(coefficients)
#
#    end
#
#end

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
    

#function next_polynomial!(γ::AbstractArray{T}, β::AbstractArray{T}, polynomials::CharacteristicPolynomials{T}, j::Int) where T<:AbstractFloat
#
#    # Generate next characteristic polynomial in the Sturm sequence.
#    # γ and β are the array of the d diagonal and subdiagonal entries at the 
#    # corresponding Jacobi matrices at index j, in the sequence.
#
#    for s in 1:length(polynomials)
#
#        next_coefficients!(polynomials.coefficients[s], γ[s], β[s], j)
#
#    end
#
#end

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
        if sign_changes(x, polynomials) >= (n - k + 1)

            z = x

        else
            y = x

        end

    end

    return x
end

#function extreme_tensorized_eigenvalues(A::KronMat{T}, char_poly::CharacteristicPolynomials{T}, k::Int) where T<:AbstractFloat
#
#
#    λ_min = 0.0
#    λ_max = 0.0 
#
#    for s in 1:length(A)
#
#        # Extract diagonal and subdiagonal entries of tridiagonal matrices Aₛ
#        pₛ = char_poly.coefficients[s]
#
#        next_coefficients!(pₛ, k, A[s][k, k], A[s][k, k-1])
#
#        y, z = initial_interval(diag(A[s], 0), diag(A[s], 1))
#
#        λ_min += bisection(y, z, k, k, pₛ)
#        λ_max += bisection(y, z, k, 1, pₛ)
#
#    end
#
#    return λ_min, λ_max
#
#end

# QR-Iterations


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

function possiblesums(eigenvalues1d::Array, d::Int)

    n    = length(eigenvalues1d)
    sums = sum.( with_replacement_combinations(eigenvalues1d, d) )
    
    return sums

end

function kronsumeigs(A::KronMat)

    tmp    = Matrix( first(A) )
    values = eigvals( tmp )

    eigenvalues = possiblesums(values, length(A))

    return eigenvalues

end

function laplace_eigenvector(n::Int, j::Int)

    vⱼ = sort([ sqrt(2 * inv(n + 1)) * sin( (i * j)π * inv(n + 1)) for i in 1:n ])

    return vⱼ

end

function laplace_eigenspace(n::Int)

    V = zeros(n, n)

    for j in 1:n

        V[:, j] = laplace_eigenvector(n, j) 

    end

    return V
end

function laplace_eigenvalue(n::Int, k::Int, j::Int)

    h = inv(n + 1)

    λⱼ = 4 * inv(h^2) * sin(j * π * inv( 2(k + 1) ) )^2

    return λⱼ

end 

function analytic_eigenvalues(d::Int, n::Int, k::Int)

    λ_min = laplace_eigenvalue(n, k, 1) * d
    λ_max = laplace_eigenvalue(n, k, k) * d

    return λ_min, λ_max

end


mutable struct SpectralData{T}

    A    ::AbstractMatrix{T}
    λ_min::Vector{T}
    λ_max::Vector{T}
    κ    ::Vector{T}
    k    ::Int      # Current iteration

    function SpectralData{T}(nmax::Int) where T

        new(zeros(nmax, nmax), fill(Inf, nmax), fill(Inf, nmax), fill(Inf, nmax), 1)

    end

    #function SpectralData{T}(A::AbstractMatrix{T}, ::Int, nmax::Int, ::LanczosUnion{T}) where T<:AbstractFloat

    #    SpectralData{T}(A, nmax)

    #end

    #function SpectralData{T}(A::AbstractMatrix{T}, d::Int, nmax::Int, ::Type{TensorArnoldi{T}}) where T<:AbstractFloat

    #    #κ = cond(eigvecs(A))^d
    #    copyA = Matrix(A)                 # There are no sparse eigenvalue solvers
    #    tmp   = minimum(eigvals(copyA)) * d

    #    λ_min = fill(tmp, nmax)
    #    

    #    new(copyA, λ_min, fill(nmax, Inf), fill(nmax, Inf), 1)

    #end


end

set_matrix!(data::SpectralData{T}, A::AbstractMatrix{T}) where T = data.A = A

Base.getindex(s::SpectralData{T}, k::Int) where T = s.λ_min[k], s.λ_max[k], s.κ[k]

function Base.setindex!(
    data     ::SpectralData{T},
    iter_data::Tuple{T, T, T},
    k        ::Int) where T  

    data[k] = iter_data

end

function current_data(data::SpectralData{T}) where T 

    k = data.k

    return data.λ_min[k], data.λ_max[k], data.κ[k]

end


Base.copy(data::SpectralData{T}) where T = SpectralData{T}(data.λ_min, data.λ_max, data.κ)

function Base.resize!(data::SpectralData{T}, k::Int) where T

    resize!(data.λ_min, k)
    resize!(data.λ_max, k)
    resize!(data.κ,     k)

end

function display(data::SpectralData{T}) where T<:AbstractFloat

    println("Spectral data: ")
    println("Smallest eigenvalues: ", data.λ_min)
    println("Largest  eigenvalues: ", data.λ_max)
    println("Condition number:     ", data.κ)

end

function Base.show(data::SpectralData) 

    display(data)

end

function update_data!(data::SpectralData{T}, d::Int, k::Int, ::LanczosUnion{T}, ::Laplace{T}) where T

    n = size(data.A, 1)

    data.λ_min[k], data.λ_max[k] = analytic_eigenvalues(d, n, k)

    data.κ[k] = data.λ_max[k] * inv(data.λ_min[k])
    data.k    = k
     
end

function getextreme(d::Int, v::AbstractArray{T}) where T

    first  = minimum(v) * d
    second = maximum(v) * d

    return first, second
end

function update_data!(data::SpectralData{T}, d::Int, k::Int, ::LanczosUnion{T}, ::RandSPD{T}) where T

    values = eigvals(@view data.A[1:k, 1:k])

    data.λ_min[k], data.λ_max[k] = getextreme(d, values)
    data.κ[k]                    = data.λ_max[k] * inv(data.λ_min[k])
    data.k                       = k

end


function update_data!(data::SpectralData{T}, d::Int,  k::Int, ::LanczosUnion{T}, ::EigValMat{T}) where T

    data.λ_min[k], data.λ_max[k] = getextreme(d, @view diag(data.A)[1:k])
    data.κ[k]                    = data.λ_max[k] * inv(data.λ_min[k])
    data.k                       = k

end

function update_data!(data::SpectralData{T}, d::Int, k::Int, ::Type{TensorArnoldi{T}}, ::ConvDiff{T}) where T


    #κ = cond(eigvecs(A))^d
    tmp = minimum(eigvals(@view data.A[1:k, 1:k])) * d

    data.λ_min[k] = tmp
    data.k = k

end
        
    
