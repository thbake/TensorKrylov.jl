using Random
using LinearAlgebra
using SparseArrays
using TensorKrylov
using Plots

abstract type Experiment{T} end 

mutable struct SPD_Experiment{T} <: Experiment{T}

    dimensions::Vector{Int}
    conv_data_vector::Vector{ConvergenceData{T}}

    function SPD_Experiment{T}(problem_dimensions::Vector{Int}, nmax::Int)

        convergence_data = [ ConvergenceData{T}(nmax) for _ in 1:length(problem_dimensions) ]

        new(problem_dimensions, convergence_data)

    end

end
    
function run_experiments(dimensions::Vector{Int}, n::Int, nmax::Int, tol::T = 1e-9) where T<:AbstractFloat

    h  = inv(n + 1)
    Aₛ = inv(h^2) * sparse(Tridiagonal(-ones(n - 1), 2ones(n), -one(n)))

    experiment = SPD_Experiment{T}(dimensions, nmax)
    
    for i in 1:length(dimensions)

        d = dimensions[i]
        A = KroneckerMatrix{T}([Aₛ for s in 1:d])
        b = [ rand(n) for _ in 1:d ]

        tensor_krylov!(experiment.conv_data_vector[i], A, b, tol, nmax) 

    end

    return experiment.conv_data_vector

end

function plot_convergence(conv_data_vector::Vector{ConvergenceData{T}})

end





