export RHSExperiment, Smoothness, Smooth, NonSmooth
export special_rhs, rhs_experiment

abstract type Smoothness end
struct Smooth    <: Smoothness end # Right hand sides are chosen so that solution of linear
struct NonSmooth <: Smoothness end # system exhibit different levels of "smoothness" 


function special_rhs(d::Int, n::Int, smoothness::Type{<:Smoothness})

    A    = assemble_matrix(n, LaplaceDense{Float64})
    Λ, Y = eigen(A)
    bs   = f(Λ, Y, smoothness)

    return [ bs for _ in 1:d ]

end

f( ::Vector, Y::Matrix, ::Type{Smooth})    = Y            * ones(size(Y, 1))
f(Λ::Vector, Y::Matrix, ::Type{NonSmooth}) = diagm(Λ) * Y * ones(size(Y, 1))

struct RHSExperiment{T} 

    experiment::Experiment{T}
    smoothness::Type{<:Smoothness}

    function RHSExperiment{T}(
        dims      ::Vector{Int},
        n         ::Int,
        nmax      ::Int,
        smoothness::Type{<:Smoothness}) where T

        rhs = [ special_rhs(d, n, smoothness) for d ∈ dims ]
        experiment = Experiment{T}(
            dims, n, nmax,
            SymInstance, 
            Laplace{T},
            TensorLanczosReorth{T},
            rhs
        )

        new(experiment, smoothness)

    end

end

function run_experiments!(rhsexp::RHSExperiment{T}, tol = 1e-9) where T

    println("Performing right-hand side experiments")

    for i in eachindex(rhsexp.experiment.dims)

        println("d = " * string(rhsexp.experiment.dims[i]))


        A = KronMat{T, rhsexp.experiment.instance}(
            rhsexp.experiment.dims[i],
            rhsexp.experiment.matrix_size,
            rhsexp.experiment.matrix_class
        )

        system = TensorizedSystem{T, rhsexp.experiment.instance}(A, rhsexp.experiment.rhs_vec[i])
        

        rhsexp.experiment.conv_vector[i] = solve_tensorized_system(
            system,
            rhsexp.experiment.nmax,
            rhsexp.experiment.orth_method,
            tol
        )

    end

end

function rhs_experiment(tol::T = 1e-9) where T

    dims = [5, 10, 50, 100]
    n    = 200
    nmax  = n - 1

    smooth_rhs    = RHSExperiment{T}(dims, n, nmax, Smooth)
    nonsmooth_rhs = RHSExperiment{T}(dims, n, nmax, NonSmooth)

    run_experiments!(smooth_rhs)
    run_experiments!(nonsmooth_rhs)

    return smooth_rhs, nonsmooth_rhs

end
