module TensorKrylov
    
    using LinearAlgebra, TensorToolbox

    # Matrix decompositions
    include("decompositions.jl")

    include("orthogonal_bases.jl")

    # Linear system solvers
    include("tensor_spd.jl")

    # Tensor approximations

    # Eigenvalue solvers
    include("eigenvalues.jl")



end # module TensorKrylov
