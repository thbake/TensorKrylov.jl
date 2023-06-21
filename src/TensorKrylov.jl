module TensorKrylov
    
    using LinearAlgebra, TensorToolbox, Kronecker

    # Matrix decompositions
    include("decompositions.jl")

    include("orthogonal_bases.jl")

    # Tensor approximations
    include("tensor_struct.jl")

    # Linear system solvers
    include("tensor_krylov_method.jl")

    # Eigenvalue solvers
    include("eigenvalues.jl")

    # Convergence bounds
    include("convergence.jl")



end # module TensorKrylov
