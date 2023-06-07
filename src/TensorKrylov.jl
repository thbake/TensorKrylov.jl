module TensorKrylov
    
    using LinearAlgebra

    include("orthogonal_bases.jl")

    # Linear system solvers
    include("tensor_spd.jl")

    # Tensor approximations

    # Eigenvalue solvers
    include("eigenvalues.jl")

    # Matrix decompositions
    include("decompositions.jl")


end # module TensorKrylov
