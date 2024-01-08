using TensorKrylov
using LinearAlgebrarepr
using SparseArrays
using DataFrames
using CSV
using Random
using Serialization

Random.seed!(12345)

abstract type Experiment{T} end

mutable struct Reproduction{T}<:Experiment{T}

    dimensions::Vector{Int}
    niterations::Int
    conv_data_vector::Vector{ConvergenceData{T}}

    function Reproduction{T}(problem_dimensions::Vector{Int}, nmax::Int) where T<:AbstractFloat

        convergence_results = [ ConvergenceData{T}(nmax) for _ in 1:length(problem_dimensions) ]

        new(problem_dimensions, nmax, convergence_results)

    end

    # For plotting, due to reliance on ssh connection for running experiments.
    function Reproduction{T}(datadir::AbstractString) where T<:AbstractFloat

        pkg_path = compute_package_directory()

        complete_path        = "experiments/data/" * datadir
        files               = cd(readdir, joinpath(pkg_path, complete_path))
        nfiles              = length(files)
        convergence_results = Vector{ConvergenceData{T}}(undef, nfiles)

        regex   = r"[0-9]+"
        matches = match.(regex, files)

        dimensions     = [ parse(Int, matches[i].match) for i in 1:length(files) ]
        sorted_indices = sortperm(dimensions)

        sorted_dims    = dimensions[sorted_indices]
        sorted_files   = files[sorted_indices]
        column_types   = [Int, T, T, T]

        for i in 1:length(sorted_files)

            df = CSV.read(
                joinpath(complete_path, sorted_files[i]),
                DataFrame,
                delim = ',',
                types = column_types
            )

            convergence_data = ConvergenceData{T}( nrow(df) )

            convergence_data.iterations              = df[:, 1]
            convergence_data.relative_residual_norm  = df[:, 2]
            convergence_data.projected_residual_norm = df[:, 3]
            convergence_data.orthogonality_data      = df[:, 4]

            convergence_results[i] = convergence_data

        end

        niterations = convergence_results[1].niterations

        new(sorted_dims, niterations, convergence_results)

    end

end

function Base.length(experiment::Reproduction{T}) where T

    return length(experiment.dimensions)

end

function get_iterations(experiment::Reproduction{T}) where T<:AbstractFloat

    return [ experiment.conv_data_vector[i].iterations for i in 1:length(experiment) ]

end

function get_relative_residuals(experiment::Reproduction{T}) where T<:AbstractFloat

    return [ experiment.conv_data_vector[i].relative_residual_norm for i in 1:length(experiment) ]

end

function get_projected_residuals(experiment::Reproduction{T}) where T<:AbstractFloat

    return [ experiment.conv_data_vector[i].projected_residual_norm for i in 1:length(experiment) ]

end

function get_convergence_data(experiment::Reproduction{T}) where T<:AbstractFloat

    return experiment.conv_data_vector

end
    

function run_experiments(
    dimensions             ::Vector{Int},
    n                      ::Int,
    nmax                   ::Int,
    matrix_instance        ::MatrixGallery,
    b                      ::Vector{KronProd{T}},
    orthonormalization_type::Type{<:TensorDecomposition{T}},
    tol                    ::T = 1e-9) where T<:AbstractFloat

    println("Performing experiments")

    systems = [ TensorizedSystem{T}(
        assemble_kronmat(dimensions[i], n, matrix_instance),
        b[i],
        orthonormalization_type) for i in eachindex(dimensions) ]
    
    experiment = Reproduction{T}(dimensions, nmax)

    for i in eachindex(dimensions)

        println("d = " * string(dimensions[i]))
        experiment.conv_data_vector[i] = solve_tensorized_system(systems[i], nmax, tol)

    end

    return experiment

end

function exportresults(exportdir::AbstractString, experiment::Reproduction{T}) where T<:AbstractFloat

    for i in 1:length(experiment)

        d    = experiment.dimensions[i]
        data = experiment.conv_data_vector[i]

        df = DataFrame(

            iterations              = data.iterations,
            relative_residual_norm  = data.relative_residual_norm,
            projected_residual_norm = data.projected_residual_norm,
            orthogonality_data      = data.orthogonality_data

        )

        output = joinpath("experiments/data/", exportdir, "data_d" * string(d) * ".csv")
        CSV.write(output, string.(df))

    end

end

function serialize_to_file(filename::AbstractString, experiment::Reproduction{T}) where T

    complete_path = "experiments/data/serialized_data/" * filename

    open(complete_path, "w") do file

        s = Serializer(file)

        serialize(s, experiment)

    end


end

function deserialize_to_file(filename::AbstractString) 

    complete_path = "experiments/data/serialized_data/" * filename

    experiment = open(complete_path, "r") do file

        s = Serializer(file)

        deserialize(s)

    end

    return experiment

        


     
end
