using DataFrames
using CSV

export bound

function compute_coefficients_directory()::AbstractString

    pkg_path    = pathof(TensorKrylov)
    regex       = r"src/TensorKrylov.jl"
    regex_match = match(regex, pkg_path)
    match_start = regex_match.offset
    directory   = pkg_path[1:match_start - 1] 

    return directory

end

function compute_dataframe(coefficients_dir::AbstractString)

    # Read csv file into dataframe
    column_types = [ fill(Float64, 64)... ]

    data = coefficients_dir * "/coefficients_data/output_data/tabelle_complete.csv"
    df   = CSV.read(data, DataFrame, delim = ',', types = column_types)

    return df

end

function compute_rank(df::DataFrame, κ::T, tol::T) where T<:AbstractFloat

    condition_order, first_digit = parse_condition(κ)

    # Find row that matches best the condition number 
    closest_row = filter(row -> row.R == first_digit * 10^condition_order, df)[:, 2:end]

    # Take ranks whose corresponding accuracy is below γ
    mask = tol .>= Vector(closest_row[1, :])

    # Extract column headers (represent tensor ranks)
    matching_ranks = parse.(Int, names(closest_row[:, mask]))

    # Take smallest rank that satisfies the bounds.
    minimum_rank = minimum(matching_ranks)

    return minimum_rank

end

function nonsymmetric_bound(λ::T, rank::Int, b_norm::T) where T<:AbstractFloat

    return 2.75 * inv(λ) * exp(-π * sqrt(rank)) * b_norm

end

function compute_rank(λ::T, b::KronProd{T}, tol::T) where T<:AbstractFloat

    b_norm = kronprodnorm(b)
    rank   = 1
    bound  = nonsymmetric_bound(λ, rank, b_norm) 

    while bound > tol

        rank += 1

        bound = nonsymmetric_bound(λ, rank, b_norm)

    end

    @info "Non-symmetric bound afer" bound

    return rank

end


function bound(λ_min::T, κ::T, b_norm::T, t::Int) where T<:AbstractFloat

    prefactor   = 16 * inv(λ_min)
    denominator = log(8 * κ)
    #t           = collect(1:63)
    nominator   = -(π^2 * t)

    #values      =  prefactor .* exp.(nominator .* inv(denominator)) .* b_norm
    #valid_ranks = t[tol .>= values]

    return prefactor * exp(nominator * inv(denominator)) * b_norm

end

function parse_condition(κ::T) where T<:AbstractFloat

    condition_order = Int(floor(log10(κ)))
    first_digit     = Int( floor(κ / (10^condition_order)) )

    return condition_order, first_digit

end

# Symmetric positive definite case
function exponential_sum_parameters(coefficients_dir::AbstractString, minimum_rank::Int, κ::T) where T<:AbstractFloat

    # Extracts coefficients αⱼ, ωⱼ > 0 and tensor rank t such that the bound of 
    # Lemma 2.6 (denoted by γ) is satisfied.
    #
    # This has been already computed for multiple pairs of t (tensor rank), and R = κ. 
    #
    # In order to satisfy the bound of Corollary 2.7, (γ / λ) ⋅ ||b̃||₂ ≤ τ, where
    # τ is the desired tolerance. This is equivalent as looking for the value
    #
    #   γ ≤ (τ ⋅ λ) / ||b̃||₂
    

    # Construct file name
    filename = coefficients_dir * "/coefficients_data/" 

    t_string = string(minimum_rank)

    if length(t_string) == 1

        filename = filename * "1_xk0" 

    else 
        
        filename = filename * "1_xk" 

    end

    condition_order, first_digit = parse_condition(κ)

    filename = filename * string(minimum_rank) * "." * string(first_digit) * "_" * string(condition_order)

    # Use three spaces to delimit the file(s)
    coeffs_df = CSV.read(
                    filename, 
                    DataFrame,
                    delim = "{",
                    header = ["number", "id"]
                )

    ω = coeffs_df[1:minimum_rank, 1]
    α = coeffs_df[minimum_rank + 1 : 2minimum_rank, 1]

    return α, ω

end

# Non-symmetric case
function exponential_sum_parameters(rank::Int)

    h_st = π * inv(sqrt(rank))

    α = [ log(exp(j * h_st) + sqrt(1 + exp(2 * j * h_st))) for j in -rank : rank ]
    ω = [ h_st * inv(sqrt((1 + exp(-2 * j * h_st))))       for j in -rank : rank ]

    return α, ω 

end
