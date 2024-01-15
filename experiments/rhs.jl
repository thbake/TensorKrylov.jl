function special_rhs(d::Int, A)

    Y = eigen(A)
    bs   = Y * ones(size(A, 1))

    return [ bs for _ in 1:d ]

end

#function run_experiments!(experiment::RHS{T}, tol = 1e-9) where T

