module TensorKrylov
    
    using LinearAlgebra, TensorToolbox, Kronecker

    # Matrix decompositions
    include("decompositions.jl")

    include("orthogonal_bases.jl")

    # Linear system solvers
    include("tensor_spd.jl")

    # Tensor approximations

    # Eigenvalue solvers
    include("eigenvalues.jl")

    # Convergence bounds
    include("convergence.jl")



end # module TensorKrylov
