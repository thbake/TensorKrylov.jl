export ConditionExperiment
export condition_experiments

mutable struct ConditionExperiment{T}

    experiment ::Experiment
    matrix_id  ::String

    function ConditionExperiment{T}(
        dims     ::Vector{Int},
        n        ::Int,
        nmax     ::Int,
        instance ::Type{<:Instance},
        class    ::Type{MatrixDep},
        orth     ::Type{<:TDecomp},
        rhs      ::rhsVec{T},
        matrix_id::String) where T

        experiment = Experiment(
            dims,
            n,
            nmax,
            instance,
            class,
            orth,
            rhs)

        new(experiment, matrix_id)

    end

end

function run_experiments!(condexp::ConditionExperiment, tol::T = 1e-9) where T

    println("Performing condition number experiments")

    for i in eachindex(condexp.experiment.dims)

        println("d = " * string(condexp.experiment.dims[i]))


        A = KronMat{condexp.experiment.instance}(
            condexp.experiment.dims[i],
            condexp.experiment.matrixsize,
            condexp.experiment.matrixclass,
            condexp.matrix_id
        )

        system = TensorizedSystem{condexp.experiment.instance}(A, condexp.experiment.rhs_vec[i])

        condexp.experiment.conv_vector[i] = solve_tensorized_system(
        system,
        condexp.experiment.nmax,
        condexp.experiment.orth_method,
        tol
        )

    end

end


function condition_experiments(::SymInstance, n::Int = 10, tol::T = 1e-9) where T

    dims = [5, 10, 50, 100]
    nmax = n
    b    = multiple_rhs(dims, n)
    b2   = multiple_rhs(dims, 13)
    b3   = multiple_rhs(dims, 20)

    hilbert_experiment = ConditionExperiment{T}(dims, n,  nmax, SymInstance, MatrixDep, TensorLanczosReorth, b, "hilb")
    pascal_experiment  = ConditionExperiment{T}(dims, 13, 13,   SymInstance, MatrixDep, TensorLanczosReorth, b2, "pascal")
    moler_experiment   = ConditionExperiment{T}(dims, 20, 20,   SymInstance, MatrixDep, TensorLanczosReorth, b3, "moler")

    run_experiments!(hilbert_experiment, tol)
    run_experiments!(pascal_experiment,  tol)
    run_experiments!(moler_experiment,   tol)

    return hilbert_experiment, pascal_experiment, moler_experiment

end

function condition_experiments(::NonSymInstance, n::Int = 10, tol::T = 1e-9) where T

    dims = [5, 10, 50, 100]
    nmax = n 
    b    = multiple_rhs(dims, n)
    b2   = multiple_rhs(dims, 13)
    b3   = multiple_rhs(dims, 20)

    hilbert_experiment = ConditionExperiment{T}(dims, n,  nmax, NonSymInstance, MatrixDep, TensorArnoldi, b, "hilb")
    pascal_experiment  = ConditionExperiment{T}(dims, 13, 13,   NonSymInstance, MatrixDep, TensorArnoldi, b2, "pascal")
    moler_experiment   = ConditionExperiment{T}(dims, 20, 20,   NonSymInstance, MatrixDep, TensorArnoldi, b3, "moler")

    run_experiments!(hilbert_experiment, tol)
    run_experiments!(pascal_experiment,  tol)
    run_experiments!(moler_experiment,   tol)

    return hilbert_experiment, pascal_experiment, moler_experiment

end
