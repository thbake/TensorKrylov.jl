export Experiment
export serialize_to_file, deserialize_from_file, get_iterations, get_relative_residuals
export ConvData, ConvVec, rhsVec, TDecomp

# Aliases
const ConvData{T} = ConvergenceData{T}
const ConvVec{T}  = Vector{ConvData{T}} 
const TDecomp     = TensorDecomposition
const rhsVec{T}   = Vector{Vector{Vector{T}}}

# Structs
abstract type AbstractExperiment end

mutable struct Experiment <: AbstractExperiment

    dims        ::Vector{Int}
    matrixsize ::Int
    nmax        ::Int
    instance    ::Type{<:Instance}
    matrixclass::Type{<:MatrixGallery}
    orth_method ::Type{<:TDecomp}
    rhs_vec     ::rhsVec{Float64}
    conv_vector::ConvVec{Float64}

    function Experiment(
        dims    ::Vector{Int},
        n       ::Int,
        nmax    ::Int,
        instance::Type{<:Instance},
        class   ::Type{<:MatrixGallery},
        orth    ::Type{<:TDecomp},
        rhs     ::rhsVec{Float64}) 

        conv_results = [ ConvData{Float64}(nmax) for _ in 1:length(dims) ]

        new(dims, n, nmax, instance, class, orth, rhs, conv_results)

    end
end

function Base.show(io::IO, experiment::Experiment) 

    println(io, "\n", typeof(experiment), " experiment:")
    println(io, "Dimensions d = ", experiment.dims, " with matrix size n = ", experiment.matrixsize,"\n")

end

Base.length(experiment::Experiment) = length(experiment.dims)

function run_experiments!(experiment::Experiment, tol::T = 1e-9) where T

    println("Performing reproduction experiments")

    for i in 1:length(experiment)

        println("d = " * string(experiment.dims[i]))

        A = KronMat{experiment.instance}(
            experiment.dims[i],
            experiment.matrixsize,
            experiment.matrixclass
        )

        system = TensorizedSystem{experiment.instance}(A, experiment.rhs_vec[i])

        experiment.conv_vector[i] = solve_tensorized_system(
            system,
            experiment.nmax,
            experiment.orth_method,
            tol
        )

    end

end

function get_iterations(experiment::Experiment) 

    return [ experiment.conv_vector[i].iterations for i in 1:length(experiment) ]

end

function get_max_iteration(experiment::Experiment)

    final_iterations = [ experiment.conv_vector[i].niterations for i in 1:length(experiment) ]

    return maximum(final_iterations)

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

compute_labels(ϵ::Vector{Float64}) = L"\epsilon = " .* string.(ϵ)

function serialize_to_file(filename::AbstractString, experiment::Experiment) 

    complete_path = "experiments/data/" * filename

    open(complete_path, "w") do file

        s = Serializer(file)

        serialize(s, experiment)

    end


end

    function deserialize_from_file(filename::AbstractString) 

    complete_path = "experiments/data/" * filename

    experiment = open(complete_path, "r") do file

        s = Serializer(file)

        deserialize(s)

    end

    return experiment
end
