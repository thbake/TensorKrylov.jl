module TensorKrylov
    
    using LinearAlgebra, TensorToolbox, Kronecker, SparseArrays, LinearSolve

    # Tensor approximations
    include("tensor_struct.jl")

    include("convergence.jl")

    include("utils.jl")

    # Preprocessing
    include("preprocessing.jl")
    
    # Matrix decompositions
    include("decompositions.jl")

    include("orthogonal_bases.jl")

    # Linear system solvers
    include("tensor_krylov_method.jl")

    #include("variants.jl")

    # Eigenvalue solvers
    include("eigenvalues.jl")

end # module TensorKrylov
