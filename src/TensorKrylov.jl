module TensorKrylov
    
    using LinearAlgebra, TensorToolbox, Kronecker, SparseArrays, LinearSolve, Logging

    # Tensor approximations
    include("tensor_struct.jl")

    include("alias.jl")

    # Matrix decompositions
    include("decompositions.jl")
    
    # Eigenvalue solvers
    include("eigenvalues.jl")

    include("convergence.jl")

    # Preprocessing
    include("preprocessing.jl")
    
    include("utils.jl")

    include("orthogonal_bases.jl")

    # Linear system solvers
    include("tensor_krylov_method.jl")

    #include("variants.jl")


end # module TensorKrylov
