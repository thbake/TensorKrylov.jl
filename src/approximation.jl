using DataFrames
using CSV

export compute_package_directory

mutable struct ApproximationData{T, U<:Instance}

    df                     ::DataFrame
    rank                   ::Int
    α                      ::Vector{T}
    ω                      ::Vector{T}
    tol                    ::T
    first_digit            ::Int
    condition_order        ::Int

    function ApproximationData{T, SymInstance}(tol::T) where T<:AbstractFloat
        package_dir = compute_package_directory()
        df          = compute_dataframe(package_dir)

        new(df, 0, zeros(1), zeros(1), tol, 0, 0) 

    end

    function ApproximationData{T, NonSymInstance}(tol::T) where T<:AbstractFloat

        new(DataFrame(), 0, zeros(1), zeros(1), tol, 0, 0)

    end

end

function compute_package_directory()::String

    pkg_path    = pathof(TensorKrylov)
    regex       = r"src/TensorKrylov.jl"
    regex_match = match(regex, pkg_path)
    match_start = regex_match.offset
    directory   = pkg_path[1:match_start - 1] 

    return directory

end

function compute_dataframe(coefficients_dir::String)

    # Read csv file into dataframe
    column_types = [ fill(Float64, 64)... ]

    data = coefficients_dir * "/coefficients_data/output_data/tabelle_complete.csv"
    df   = CSV.read(data, DataFrame, delim = ',', types = column_types)

    return df

end

function getclosestrow(df::DataFrame, condition_order::Int, first_digit::Int)

    # Find row that matches best the condition number 
    closest_row = filter(row -> row.R == first_digit * 10.0^condition_order, df)[:, 2:end]

    return closest_row

end

function compute_rank!(approxdata::ApproximationData{T, SymInstance}, κ::T) where T<:AbstractFloat

    approxdata.condition_order, approxdata.first_digit = parse_condition(κ)

    closest_row = getclosestrow(approxdata.df, approxdata.condition_order, approxdata.first_digit)

    while nrow(closest_row) == 0

        approxdata.first_digit += 1
        closest_row  = getclosestrow(approxdata.df, approxdata.condition_order, approxdata.first_digit)

    end

    # Take ranks whose corresponding accuracy is below γ
    mask            = approxdata.tol .>= Vector(closest_row[1, :])
    matching_ranks  = parse.(Int, names(closest_row[:, mask])) # Extract column headers (represent tensor ranks)
    minimum_rank    = minimum(matching_ranks) # Take smallest rank that satisfies the bounds.
    approxdata.rank = minimum_rank

end

nonsymmetric_bound(λ::T, rank::Int) where T = 2.75 * inv(λ) * exp(-π * sqrt(rank / 2))

function compute_rank!(
    approxdata  ::ApproximationData{T, NonSymInstance},
    spectraldata::SpectralData{matT, T, NonSymInstance}) where {matT, T<:AbstractFloat}

    λ_min, _, _ = current_data(spectraldata)

    rank   = 1
    bound  = nonsymmetric_bound(λ_min, rank) 

    while bound > approxdata.tol

        rank += 1

        bound = nonsymmetric_bound(λ_min, rank)

    end

    approxdata.rank = rank

end

function parse_condition(κ::T) where T<:AbstractFloat

    condition_order = Int(floor(log10(κ)))
    first_digit     = Int( floor(κ / (10^float(condition_order))) )

    return condition_order, first_digit

end

# Symmetric positive definite case
function exponential_sum_parameters!(approxdata::ApproximationData{T, SymInstance}, coefficients_dir::AbstractString) where T<:AbstractFloat

    # Extracts coefficients αⱼ, ωⱼ > 0 and tensor rank t such that the bound of 
    # Lemma 2.6 (denoted by γ) is satisfied.
    #
    # This has been already computed for multiple pairs of t (tensor rank), and R = κ. 
    

    # Construct file name
    filename = coefficients_dir * "coefficients_data/" 

    t_string = string(approxdata.rank)

    length(t_string) == 1 ? filename *= "1_xk0" : filename *= "1_xk"

    filename *= t_string * "." * string(approxdata.first_digit) * "_" * string(approxdata.condition_order)

    # Use three spaces to delimit the file(s)
    coeffs_df = CSV.read(
                    filename, 
                    DataFrame,
                    delim = "{",
                    header = ["number", "id"]
                )

    approxdata.ω = coeffs_df[1:approxdata.rank, 1]
    approxdata.α = coeffs_df[approxdata.rank + 1 : 2approxdata.rank, 1]

end

# Non-symmetric case
function exponential_sum_parameters!(data::ApproximationData{T, NonSymInstance}) where T<:AbstractFloat

    t = data.rank

    h_st   = π * inv(sqrt(t))
    data.α = [ log(exp(j * h_st) + sqrt(1 + exp(2 * j * h_st) ) ) for j in -t : t ]
    data.ω = [ h_st * inv( sqrt( (1 + exp(-2 * j * h_st) ) ) )    for j in -t : t ]

end

function update_data!(approxdata::ApproximationData{T, SymInstance}, spectraldata::SpectralData{matT, T, SymInstance}) where {matT, T<:AbstractFloat}

    package_dir                = compute_package_directory()
    approxdata.df              = compute_dataframe(package_dir)
    _, _, κ                    = current_data(spectraldata)
    compute_rank!(approxdata, κ)
    exponential_sum_parameters!(approxdata, package_dir)

end

function update_data!(approxdata::ApproximationData{T, NonSymInstance}, spectraldata::SpectralData{matT, T, NonSymInstance}) where {matT, T<:AbstractFloat}

    compute_rank!(approxdata, spectraldata)
    exponential_sum_parameters!(approxdata)

end
