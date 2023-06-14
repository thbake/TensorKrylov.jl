function spd_error_bound(
        A::KroneckerMatrix{T},
        b::AbstractVector,
        λ,
        κ::Vector{T})::T where T<:AbstractFloat

    # Compute the 2-norm  of A
    #max_singular_values = [ norm(A[s]) for s in eachindex(A) ]
    
    A_norm = norm(A)

    b_norm = prod( map(norm, b) )



end
