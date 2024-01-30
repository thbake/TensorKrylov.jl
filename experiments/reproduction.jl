using Random 

export reproduce

Random.seed!(1234)

struct Reproduction end     

function reproduce(tol::T = 1e-9) where T

    dims = [5, 10, 50, 100]
    n    = 200
    nmax = 199
    b    = multiple_rhs(dims, n)
    
    spd    = Experiment{T}(dims, n, nmax, SymInstance,    Laplace{T},  TensorLanczosReorth{T}, b)
    nonsym = Experiment{T}(dims, n, nmax, NonSymInstance, ConvDiff{T}, TensorArnoldi{T}, b)

    run_experiments!(spd,    tol)
    run_experiments!(nonsym, tol)

    return spd, nonsym

end
