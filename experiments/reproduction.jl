using Random 

export reproduce

Random.seed!(1234)

struct Reproduction end     

function reproduce(n::Int = 200, tol::T = 1e-9) where T

    dims = [5, 10, 50, 100]
    nmax = n 
    b    = multiple_rhs(dims, n)
    
    spd    = Experiment(dims, n, nmax, SymInstance,    Laplace,  TensorLanczosReorth, b)
    nonsym = Experiment(dims, n, nmax, NonSymInstance, ConvDiff, TensorArnoldi,       b)

    run_experiments!(spd,    tol)
    run_experiments!(nonsym, tol)

    return spd, nonsym

end
