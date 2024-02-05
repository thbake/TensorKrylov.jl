export EigValDist
export clusterone, clusterzero

mutable struct EigValDist{T} <: AbstractExperiment{T}

    experiment ::Experiment{T}
    eigenvalues::Vector{T}

    function EigValDist{T}(
        dims       ::Vector{Int},
        eigenvalues::Vector{T},
        nmax       ::Int,
        rhs        ::rhsVec{T}) where T

        instance   = SymInstance
        experiment = Experiment{T}(
            dims,
            length(eigenvalues),
            nmax,
            instance,
            EigValMat{T},
            TensorLanczosReorth{T},
            rhs)

        new(experiment, eigenvalues)

    end

end


function run_experiments!(distexp::EigValDist{T}, tol = 1e-9) where T

    println("Performing eigenvalue experiments")

    for i in eachindex(distexp.experiment.dims)

        println("d = " * string(distexp.experiment.dims[i]))


        A = KronMat{T, distexp.experiment.instance}(
            distexp.experiment.dims[i],
            distexp.eigenvalues,
            distexp.experiment.matrix_class
        )

        system = TensorizedSystem{T, distexp.experiment.instance}(A, distexp.experiment.rhs_vec[i])
        

        distexp.experiment.conv_vector[i] = solve_tensorized_system(
            system,
            distexp.experiment.nmax,
            distexp.experiment.orth_method,
            tol
        )

    end

end

clusterzero(n::Int) = [j^2 * inv(400) for j in 1:n]

function clusterone(n::Int)

    values    = zeros(n)
    values[1] = inv(400)

    tmp = log(n)

    for j in 2:n

        values[j] = log(j) * inv(tmp) 

    end

    return values

end



