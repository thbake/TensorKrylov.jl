module TensorKrylov
    
    using LinearAlgebra, TensorToolbox, Kronecker, SparseArrays, Logging
    import LinearAlgebra: norm, mul!

    # Tensor approximations
    include("tensor_struct.jl")

    include("alias.jl")

    # Matrix/tensor decompositions
    include("decompositions.jl")
    
    # Generation of orthonormal Krylov subspace bases
    include("orthogonal_bases.jl")

    # Eigenvalue solvers
    include("eigenvalues.jl")

    include("convergence.jl")

    # Preprocessing
    include("preprocessing.jl")
    
    include("utils.jl")

    # Linear system solvers
    include("tensor_krylov_method.jl")

    #include("variants.jl")


end # module TensorKrylov
