using Random
using LinearAlgebra
using SparseArrays
using TensorKrylov
using Plots
using DataFrames
using CSV

abstract type Problem end
struct SPDProblem    <: Problem end
struct NonSymProblem <: Problem end

mutable struct Experiment{T} 

    dimensions::Vector{Int}
    conv_data_vector::Vector{ConvergenceData{T}}

    function Experiment{T}(problem_dimensions::Vector{Int}, nmax::Int) where T<:AbstractFloat

        convergence_results = [ ConvergenceData{T}(nmax) for _ in 1:length(problem_dimensions) ]

        new(problem_dimensions, convergence_results)

    end

    function Experiment{T}(datadir::AbstractString = "experiments/spd_data") where T<:AbstractFloat

        files               = cd(readdir, datadir)
        nfiles              = length(files)
        problem_dimensions  = Vector{Int}(undef, nfiles)
        convergence_results = Vector{ConvergenceData{T}}(undef, nfiles)

        regex   = r"[0-9]+"
        matches = match.(regex, files)

        dimensions     = [ parse(Int, matches[i].match) for i in 1:length(files) ]
        sorted_indices = sortperm(dimensions)

        sorted_dims    = dimensions[sorted_indices]
        sorted_files   = files[sorted_indices]
        column_types   = [Int, T, T]

        for i in 1:length(sorted_files)

            df = CSV.read(
                joinpath(datadir, sorted_files[i]),
                DataFrame,
                delim = ',',
                types = column_types
            )

            convergence_data = ConvergenceData{T}( nrow(df) )

            convergence_data.iterations              = df[:, 1]
            convergence_data.relative_residual_norm  = df[:, 2]
            convergence_data.projected_residual_norm = df[:, 3]

            convergence_results[i] = convergence_data

        end

        new(sorted_dims, convergence_results)

    end

end

function Base.length(experiment::Experiment{T}) where T

    return length(experiment.dimensions)

end

function get_iterations(experiment::Experiment{T}) where T<:AbstractFloat

    return [ experiment.conv_data_vector[i].iterations for i in 1:length(experiment) ]

end

function get_relative_residuals(experiment::Experiment{T}) where T<:AbstractFloat

    return [ experiment.conv_data_vector[i].relative_residual_norm for i in 1:length(experiment) ]

end
    
function assemble_matrix(n::Int, ::Type{SPDProblem})

    h  = inv(n + 1)
    Aₛ = inv(h^2) * sparse(SymTridiagonal(2ones(n), -ones(n)))

    return Aₛ

end

function assemble_matrix(n::Int, ::Type{NonSymProblem}, c::AbstractFloat = 10.0) 

    h  = inv(n + 1)
    L  =     sparse( inv(h^2) .* SymTridiagonal( 2ones(n), -ones(n - 1)) )
    Aₛ = L + sparse( (c * inv(4 * h)) .* diagm(-1 => ones(n-1), 0 => 3ones(n), 1 => -5ones(n - 1), 2 => ones(n - 2)) )

    return Aₛ

end

function get_orthonormalization(::SPDProblem) 

    return TensorLanczos

end

function get_orthonormalization(::NonSymProblem) 

    return TensorArnoldi

end

function run_experiments(dimensions::Vector{Int}, n::Int, nmax::Int, problem::Type{Problem}, tol::T = 1e-9, normalize_rhs::Bool = true) where T<:AbstractFloat

    Aₛ = assemble_matrix(n, problem)

    experiment = Experiment{T}(dimensions, nmax)
    
    for i in 1:length(dimensions)

        d = dimensions[i]
        A = KroneckerMatrix{T}([Aₛ for s in 1:d])
        b = [ rand(n) for _ in 1:d ]

        if normalize_rhs

            normalize!(b)

        end

        orthonormalization_type = get_orthonormalization(problem)

        tensor_krylov!(experiment.conv_data_vector[i], A, b, tol, nmax, orthonormalization_type) 

    end

    return experiment

end

function exportresults(exportdir::AbstractString, experiment::Experiment{T}) where T<:AbstractFloat

    for i in 1:length(experiment)

        d    = experiment.dimensions[i]
        data = experiment.conv_data_vector[i]

        df = DataFrame(

            data.iterations,
            data.relative_residual_norm,
            data.projected_residual_norm,

        )

        output = joinpath(exportdir, "data_d" * string(d) * ".csv")
        CSV.write(output, string.(df))

    end

end

function plotexperiment(experiment::Experiment{T}) where T<:AbstractFloat

    labels         = "d = " .* string.(experiment.dimensions)
    total_iterations         = get_iterations(experiment)
    total_relative_residuals = get_relative_residuals(experiment)

    plot(total_iterations, total_relative_residuals, labels = permutedims(labels))

    plot!(yscale=:log10)
    ylims!(1e-8, 1e+2)
    xlabel!("k")
    ylabel!("Relative residual norm")

end
