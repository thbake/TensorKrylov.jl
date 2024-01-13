include("experiment_common.jl")

mutable struct Reproduction{T} <: Experiment{T}

    dimensions  ::Vector{Int}
    matrix_size ::Int
    nmax        ::Int
    instance    ::Type{<:Instance}
    matrix_class::Type{<:MatrixGallery{T}}
    orth_method ::Type{<:TDecomp{T}}
    rhs_vec     ::rhsVec{T}
    conv_vector::ConvVec{T}

    function Reproduction{T}(
        dims    ::Vector{Int},
        n       ::Int,
        nmax    ::Int,
        instance::Type{<:Instance},
        class   ::Type{<:MatrixGallery{T}},
        orth    ::Type{<:TDecomp{T}},
        rhs     ::rhsVec{T}) where T

        conv_results = [ ConvData{T}(nmax, instance) for _ in 1:length(dims) ]

        new(dims, n, nmax, instance, class, orth, rhs, conv_results)

    end

    # For plotting, due to reliance on ssh connection for running experiments.
    function Reproduction{T}(datadir::AbstractString) where T<:AbstractFloat

        pkg_path = compute_package_directory()

        complete_path        = "experiments/data/" * datadir
        files               = cd(readdir, joinpath(pkg_path, complete_path))
        nfiles              = length(files)
        convergence_results = Vector{ConvData{T}}(undef, nfiles)

        regex   = r"[0-9]+"
        matches = match.(regex, files)

        dimensions     = [ parse(Int, matches[i].match) for i in 1:length(files) ]
        sorted_indices = sortperm(dimensions)

        sorted_dims    = dimensions[sorted_indices]
        sorted_files   = files[sorted_indices]
        column_types   = [Int, T, T, T]

        for i in 1:length(sorted_files)

            df = CSV.read(
                joinpath(complete_path, sorted_files[i]),
                DataFrame,
                delim = ',',
                types = column_types
            )

            convergence_data = ConvData{T}( nrow(df) )

            convergence_data.iterations              = df[:, 1]
            convergence_data.relative_residual_norm  = df[:, 2]
            convergence_data.projected_residual_norm = df[:, 3]
            convergence_data.orthogonality_data      = df[:, 4]

            convergence_results[i] = convergence_data

        end

        niterations = convergence_results[1].niterations

        new(sorted_dims, niterations, convergence_results)

    end

end

function run_experiments!(experiment::Reproduction{T}, tol::T = 1e-9) where T

    println("Performing reproduction experiments")

    for i in eachindex(experiment.dimensions)

        println("d = " * string(experiment.dimensions[i]))

        A = KronMat{T, experiment.instance}(
            experiment.dimensions[i],
            experiment.matrix_size,
            experiment.matrix_class
        )

        system = TensorizedSystem{T, experiment.instance}(A, experiment.rhs_vec[i])

        experiment.conv_vector[i] = solve_tensorized_system(
            system,
            experiment.nmax,
            experiment.orth_method,
            tol
        )

    end

end

function get_relative_residuals(experiment::Reproduction) 

    return [ experiment.conv_vector[i].relative_residual_norm for i in 1:length(experiment) ]

end

function get_projected_residuals(experiment::Reproduction) 

    return [ experiment.conv_vector[i].projected_residual_norm for i in 1:length(experiment) ]

end

function get_convergence_data(experiment::Reproduction) 

    return experiment.conv_vector

end
    


function exportresults(exportdir::AbstractString, experiment::Reproduction{T}) where T<:AbstractFloat

    for i in 1:length(experiment)

        d    = experiment.dimensions[i]
        data = experiment.conv_vector[i]

        df = DataFrame(

            iterations              = data.iterations,
            relative_residual_norm  = data.relative_residual_norm,
            projected_residual_norm = data.projected_residual_norm,
            orthogonality_data      = data.orthogonality_data

        )

        output = joinpath("experiments/data/", exportdir, "data_d" * string(d) * ".csv")
        CSV.write(output, string.(df))

    end

end

