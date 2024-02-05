export Experiment
export serialize_to_file, deserialize_from_file, get_iterations

# Aliases
const ConvData{T} = ConvergenceData{T}
const ConvVec{T}  = Vector{ConvData{T}} 
const TDecomp{T}  = TensorDecomposition{T}
const rhsVec{T}   = Vector{Vector{Vector{T}}}

# Structs
abstract type AbstractExperiment{T} end

mutable struct Experiment{T} <: AbstractExperiment{T}

    dims        ::Vector{Int}
    matrix_size ::Int
    nmax        ::Int
    instance    ::Type{<:Instance}
    matrix_class::Type{<:MatrixGallery{T}}
    orth_method ::Type{<:TDecomp{T}}
    rhs_vec     ::rhsVec{T}
    conv_vector::ConvVec{T}

    function Experiment{T}(
        dims    ::Vector{Int},
        n       ::Int,
        nmax    ::Int,
        instance::Type{<:Instance},
        class   ::Type{<:MatrixGallery{T}},
        orth    ::Type{<:TDecomp{T}},
        rhs     ::rhsVec{T}) where T

        conv_results = [ ConvData{T}(nmax, instance) for _ in 1:length(dims) ]

        new(dims, n, nmax, instance, class, orth, rhs, conv_results)

    end
end

function Base.show(io::IO, experiment::Experiment{T}) where T

    println(io, "\n", typeof(experiment), " experiment:")
    println(io, "Dimensions d = ", experiment.dims, " with matrix size n = ", experiment.matrix_size,"\n")

end

Base.length(experiment::Experiment{T}) where T = length(experiment.dims)

function run_experiments!(experiment::Experiment{T}, tol::T = 1e-9) where T

    println("Performing reproduction experiments")

    for i in 1:length(experiment)

        println("d = " * string(experiment.dims[i]))

        A = KronMat{T, experiment.instance}(
            experiment.dims[i],
            experiment.matrix_size,
            experiment.matrix_class
        )

        system = TensorizedSystem{T, experiment.instance}(A, experiment.rhs_vec[i])

        experiment.conv_vector[i] = solve_tensorized_system(
            system,
            experiment.nmax,
            experiment.orth_method,
            tol
        )

    end

end

function get_iterations(experiment::Experiment{T}) where T

    return [ experiment.conv_vector[i].iterations for i in 1:length(experiment) ]

end

function get_relative_residuals(experiment::Experiment) 

    return [ experiment.conv_vector[i].relative_residual_norm for i in 1:length(experiment) ]

end

function get_projected_residuals(experiment::Experiment) 

    return [ experiment.conv_vector[i].projected_residual_norm for i in 1:length(experiment) ]

end

function get_convergence_data(experiment::Experiment) 

    return experiment.conv_vector

end

compute_labels(experiment::Experiment) =  "dim = " .* string.(experiment.dims)

compute_labels(dims::Vector{Int}) = "dim = " .* string.(dims)

function serialize_to_file(filename::AbstractString, experiment::Experiment{T}) where T

    complete_path = "experiments/data/reproduction_data/" * filename

    open(complete_path, "w") do file

        s = Serializer(file)

        serialize(s, experiment)

    end


end

    function deserialize_from_file(filename::AbstractString) 

    complete_path = "experiments/data/reproduction_data/" * filename

    experiment = open(complete_path, "r") do file

        s = Serializer(file)

        deserialize(s)

    end

    return experiment
end
