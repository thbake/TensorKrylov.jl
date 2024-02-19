module TensorKrylov
    
    using LinearAlgebra, Kronecker, SparseArrays, Logging
    import LinearAlgebra: norm, mul!

    # Tensor structures representing componets of linear systems
    include("tensor_struct.jl")

    include("alias.jl")

    # Matrix/tensor decompositions
    include("decompositions.jl")
    
    # Generation of orthonormal Krylov subspace bases
    include("orthogonal_bases.jl")

    # Eigenvalue solvers
    include("eigenvalues.jl")

    # Convergence data structures
    include("convergence.jl")

    # Preprocessing
    include("approximation.jl")
    
    # Utilities
    include("utils.jl")

    # Linear system solvers
    include("tensor_krylov_method.jl")

    # Assembly of linear system
    include("system.jl")

    #include("variants.jl")


end # module TensorKrylov
