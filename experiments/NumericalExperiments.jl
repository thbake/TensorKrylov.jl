module NumericalExperiments

    using TensorKrylov
    using CSV, DataFrames, LaTeXStrings, LinearAlgebra, Plots, Serialization, SparseArrays 

    export run_experiments!

    # General interface
    include("experiment_common.jl")

    # Experiments
    include("reproduction.jl")

    include("eigenvalue_distribution.jl")

    include("rhs.jl")

    include("ill_conditioned.jl")

    include("parametrized_systems.jl")

    # Plots
    include("plot_general.jl")
    
    include("plot_eigenvalues.jl")

end
