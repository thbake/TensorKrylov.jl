using Random 
export EigValDist
export clusterone, clusterzero, eigenvalue_experiment

Random.seed!(1234)

mutable struct EigValDist{T}

    experiment ::Experiment
    eigenvalues::Vector{T}

    function EigValDist{T}(
        dims       ::Vector{Int},
        eigenvalues::Vector{T},
        nmax       ::Int,
        rhs        ::rhsVec{T}) where T

        instance   = SymInstance
        experiment = Experiment(
            dims,
            length(eigenvalues),
            nmax,
            instance,
            EigValMat,
            TensorLanczosReorth,
            rhs)

        new(experiment, eigenvalues)

    end

end


function run_experiments!(distexp::EigValDist{T}, tol = 1e-9) where T

    println("Performing eigenvalue experiments")

    for i in eachindex(distexp.experiment.dims)

        println("d = " * string(distexp.experiment.dims[i]))


        A = KronMat{distexp.experiment.instance}(
            distexp.experiment.dims[i],
            distexp.eigenvalues,
            distexp.experiment.matrix_class
        )

        system = TensorizedSystem{distexp.experiment.instance}(A, distexp.experiment.rhs_vec[i])
        

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

function eigenvalue_experiment(n::Int = 15, tol::T = 1e-9) where T

    dims = [5, 10, 50, 100]
    nmax = n 
    b    = multiple_rhs(dims, n)
    
    eigszero = clusterzero(n)
    eigsone  = clusterone(n)

    distzero = EigValDist{T}(dims, eigszero, nmax, b)
    distone  = EigValDist{T}(dims, eigsone,  nmax, b)

    run_experiments!(distzero, tol)
    run_experiments!(distone,  tol)

    return distzero, distone

end

