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

end

function Base.length(experiment::Experiment{T}) where T

    return length(experiment.dimensions)

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

function get_orthonormalization(::SPDProblem) where T<:AbstractFloat

    return TensorLanczos{T}

end

function get_orthonormalization(::NonSymProblem) where T<:AbstractFloat

    return TensorArnoldi{T}

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
