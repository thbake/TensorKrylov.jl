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

function possible_sums(eigenvalues, d::Int, ϵ::Float64)

    result          = perturb_eigenvalues(eigenvalues, d, ϵ)
    tmp             = [ row for row in eachrow(result) ]

    return reshape(sum.(collect(Iterators.product(tmp...))), length(eigenvalues)^d)

end

function perturb_eigenvalues(eigenvalues, d::Int, ϵ::Float64)

    n = length(eigenvalues)

    result = zeros(d, n)

    for s in 1:d

        result[s, :] = (s * ϵ) .+ eigenvalues

    end

    return result

end

function perturb_matrix!(A::KronMat, ϵ::Float64)

    d = length(A)

    for s in 1:d

        A[s] = (s * ϵ) .+ A[s] 

    end

end


function run_experiments!(distexp::EigValDist{T}, ϵ::T = 0.0, tol = 1e-9) where T

    println("Performing eigenvalue experiments")

    for i in eachindex(distexp.experiment.dims)

        println("d = " * string(distexp.experiment.dims[i]))


        A = KronMat{distexp.experiment.instance}(
            distexp.experiment.dims[i],
            distexp.eigenvalues,
            distexp.experiment.matrixclass
        )

        perturb_matrix!(A, ϵ)

        system = TensorizedSystem{distexp.experiment.instance}(A, distexp.experiment.rhs_vec[i])
        

        distexp.experiment.conv_vector[i] = solve_tensorized_system(
            system,
            distexp.experiment.nmax,
            distexp.experiment.orth_method,
            tol
        )

    end

end

function clusterzero(n::Int)

    κ = n^2 

    return [ j^2 * inv(κ) for j in 1:n ]

end

function clusterone(n::Int)

    values    = zeros(n)
    values[1] = inv(n^2)

    tmp = log(n)

    for j in 2:n

        values[j] = log(j) * inv(tmp) 

    end

    return values

end

function eigenvalue_experiment(n::Int, b, ϵ::T = 0.0, tol::T = 1e-9) where T

    dims = [5, 10, 50, 100]
    nmax = n 
    
    eigszero = clusterzero(n)
    eigsone  = clusterone(n)

    distzero = EigValDist{T}(dims, eigszero, nmax, b)
    distone  = EigValDist{T}(dims, eigsone,  nmax, b)

    run_experiments!(distzero, ϵ, tol)
    run_experiments!(distone,  ϵ, tol)

    return distzero, distone

end

