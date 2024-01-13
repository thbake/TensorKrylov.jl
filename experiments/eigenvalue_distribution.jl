include("experiment_common.jl")

mutable struct EigValsDist{T} <: Experiment{T}
    dimensions ::Vector{Int}
    matrix_size::Int
    nmax       ::Int
    rhs_vec    ::rhsVec{T}
    conv_vector::ConvVec{T}
    eigenvalues::Vector{T}

    function EigValsDist{T}(
        dims       ::Vector{Int},
        eigenvalues::Vector{T},
        nmax       ::Int,
        rhs        ::rhsVec{T}) where T

        conv_results = [ ConvData{T}(nmax, SymInstance) for _ in 1:length(dims) ]

            new(dims, length(eigenvalues), nmax, rhs, conv_results, eigenvalues)

    end

end

function run_experiments!(experiment::EigValsDist{T}, tol = 1e-9) where T

    println("Performing eigenvalue experiments")

    for i in eachindex(experiment.dimensions)

        println("d = " * string(experiment.dimensions[i]))

        A = KronMat{T, SymInstance}(
            experiment.dimensions[i],
            experiment.eigenvalues,
            EigValMat{T}
        )

        system = TensorizedSystem{T, SymInstance}(A, experiment.rhs_vec[i])

        experiment.conv_vector[i] = solve_tensorized_system(
            system,
            experiment.nmax,
            TensorLanczosReorth{T},
            tol
        )

    end

end

clusterzero(n::Int) = [j^2 * inv(400) for j in 1:n]

function clusterone(n::Int)

    values    = zeros(n)
    values[1] = 1 * inv(400)

    tmp = log(n)

    for j in 2:n

        values[j] = log(j) * inv(tmp) 

    end

    return values

end

