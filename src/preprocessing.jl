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

function bound(λ_min::T, κ::T, b_norm::T, t::Int) where T<:AbstractFloat

    prefactor   = 16 * inv(λ_min)
    denominator = log(8 * κ)
    #t           = collect(1:63)
    nominator   = -(π^2 * t)

    #values      =  prefactor .* exp.(nominator .* inv(denominator)) .* b_norm
    #valid_ranks = t[tol .>= values]

    return prefactor * exp(nominator * inv(denominator)) * b_norm

end

function optimal_coefficients(coefficients_dir::AbstractString, df::DataFrame, τ::T, κ::T, λ::T, b̃_norm::T) where T<:AbstractFloat

    # Extracts coefficients αⱼ, ωⱼ > 0 and tensor rank t such that the bound of 
    # Lemma 2.6 (denoted by γ) is satisfied.
    #
    # This has been already computed for multiple pairs of t (tensor rank), and R = κ. 
    #
    # In order to satisfy the bound of Corollary 2.7, (γ / λ) ⋅ ||b̃||₂ ≤ τ, where
    # τ is the desired tolerance. This is equivalent as looking for the value
    #
    #   γ ≤ (τ ⋅ λ) / ||b̃||₂
    #

    # Compute desired tolerance
    γ = λ * inv(b̃_norm) * τ

    # Compute order
    condition_order = Int(floor(log10(κ)))

    first_digit = Int( floor(κ / (10^condition_order)) )

    # Find row that matches best the condition number 
    closest_row = filter(row -> row.R == first_digit * 10^condition_order, df)[:, 2:end]

    # Take ranks whose corresponding accuracy is below γ
    mask = τ .>= Vector(closest_row[1, :])

    masked_row = Vector(closest_row[1, :])[mask]

    # Extract column headers (represent tensor ranks)
    matching_ranks = parse.(Int, names(closest_row[:, mask]))

    # Take smallest rank that satisfies the bounds.
    t_min = minimum(matching_ranks)

    # Construct file name
    filename = coefficients_dir * "/coefficients_data/" 

    t_string = string(t_min)

    if length(t_string) == 1

        filename = filename * "1_xk0" 

    else 
        
        filename = filename * "1_xk" 

    end

    filename = filename * string(t_min) * "." * string(first_digit) * "_" * string(condition_order)

    # Use three spaces to delimit the file(s)
    coeffs_df = CSV.read(
                    filename, 
                    DataFrame,
                    delim = "{",
                    header = ["number", "id"]
                )

    ω = coeffs_df[1:t_min, 1]
    α = coeffs_df[t_min + 1 : 2t_min, 1]

    return ω, α, t_min

end

function nonsymmetric_bound(β::T, γ::T, κ::T, rank::Int, b_norm::T) where T<:AbstractFloat

    #return κ * inv(β) * exp(γ * π) * exp(-π * sqrt(rank)) * b_norm
    return inv(β) * exp(γ * π) * exp(-π * sqrt(rank)) 

end

function get_nonsymmetric_rank(A::AbstractMatrix{T}, b::KronProd{T}, tol::T) where T<:AbstractFloat

    d    = length(b)
    Λ, P = eigen(Matrix(A))
    κ    = cond(P)^d

    β = minimum(real.(Λ))
    γ = maximum(imag.(Λ))

    b_norm = kronprodnorm(b)

    rank  = 1
    bound = nonsymmetric_bound(β, γ, κ, rank, b_norm) 

    while bound > tol

        rank += 1

        bound = nonsymmetric_bound(β, γ, κ, rank, b_norm)

    end

    return rank

end

function nonsymmetric_coefficients(rank::Int)

    h_st = π * inv(rank)

    α = [ log(exp(j * h_st) + sqrt(1 + exp(2 * j * h_st))) for j in -rank : rank ]
    ω = [ h_st * inv(sqrt((1 + exp(-2 * j * h_st))))       for j in -rank : rank ]

    return α, ω 

end

#function approximate_minreciprocal(α::Array{T}, ω::Array{T}, )
