include("reproduction_spd.jl")

function special_rhs(d::Int, A)

    Y = eigen(A)
    bs   = Y * ones(size(A, 1))

    return [ bs for _ in 1:d ]

end
