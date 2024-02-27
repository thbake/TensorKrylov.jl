export parametrize, parametrized_experiment

function parametrize(α::T, ::SymInstance) where T 

    n = 200
    h = inv(n + 1)
    A = sparse(SymTridiagonal(α * ones(n), -ones(n - 1)))

    return A
end

function parametrize(β::T, ::NonSymInstance) where T 

    n  = 200
    h  = inv(n + 1)
    L  = sparse( inv(h^2) .* SymTridiagonal( 2ones(n), -ones(n-1)) )
    A = sparse( L + (10 * inv(4 * h)) .* diagm(-1 => ones(n-1), 0 => 3ones(n), 1 => -β * ones(n-1), 2 => ones(n-2)) )

    return A
end

function run_experiments!(experiment::Experiment, parameter::T, tol::T = 1e-9) where T

    println("Performing reproduction experiments")

    Aₛ = parametrize(parameter, experiment.instance()) # Initialize coefficient matrix

    for i in 1:length(experiment)

        println("d = " * string(experiment.dims[i]))

        A              = KronMat{experiment.instance}(Aₛ, experiment.dims[i])
        A.matrixclass = experiment.matrixclass
        system = TensorizedSystem{experiment.instance}(A, experiment.rhs_vec[i])

        experiment.conv_vector[i] = solve_tensorized_system(
            system,
            experiment.nmax,
            experiment.orth_method,
            tol
        )

    end
end

function parametrized_experiment(α::T, β::T, tol::T = 1e-9) where T

    dims = [5, 10, 50, 100]
    n    = 200
    nmax = n 
    b    = multiple_rhs(dims, n)
    
    spd    = Experiment(dims, n, nmax, SymInstance,    Laplace,  TensorLanczosReorth, b)
    nonsym = Experiment(dims, n, nmax, NonSymInstance, ConvDiff, TensorArnoldi,       b)

    run_experiments!(spd,    α, tol)
    run_experiments!(nonsym, β, tol)

    return spd, nonsym

end
