using TensorKrylov
using LinearAlgebra
using SparseArrays
using DataFrames
using CSV
using Random
using Serialization

# Aliases
const ConvData{T} = ConvergenceData{T}
const ConvVec{T}  = Vector{ConvData{T}} 
const TDecomp{T}  = TensorDecomposition{T}
const rhsVec{T}   = Vector{Vector{Vector{T}}}

# Structs
abstract type Experiment{T} end

function Base.show(io::IO, experiment::Experiment{T}) where T

    println(io, "\n", typeof(experiment), " experiment:")
    println(io, "Dimensions d = ", experiment.dimensions, " with matrix size n = ", experiment.matrix_size,"\n")

end

function Base.length(experiment::Experiment{T}) where T

    return length(experiment.dimensions)

end

function get_iterations(experiment::Experiment{T}) where T

    return [ experiment.conv_vector[i].iterations for i in 1:length(experiment) ]

end

function serialize_to_file(filename::AbstractString, experiment::Experiment{T}) where T

    complete_path = "experiments/data/" * filename

    open(complete_path, "w") do file

        s = Serializer(file)

        serialize(s, experiment)

    end


end

function deserialize_to_file(filename::AbstractString) 

    complete_path = "experiments/data/" * filename

    experiment = open(complete_path, "r") do file

        s = Serializer(file)

        deserialize(s)

    end

    return experiment
end
