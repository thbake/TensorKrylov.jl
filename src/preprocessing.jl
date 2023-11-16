using DataFrames
using CSV

function compute_dataframe()

    # Read csv file into dataframe
    #data = "../coefficients_data/output_data/k_R_accuracy_sorted.csv"
    data = "../coefficients_data/output_data/tabelle_complete.csv"

    column_types = [ fill(Float64, 64)... ]

    df = CSV.read(data, DataFrame, delim = ',', types = column_types)

    return df

end

function bound(λ_min::T, κ::T, b_norm::T, tol::T) where T<:AbstractFloat

    prefactor   = 16 * inv(λ_min)
    denominator = log(8 * κ)
    t           = collect(1:63)
    nominator   = -π^2 .* t 
    
    values =  prefactor .* exp.(nominator .* inv(denominator)) .* b_norm

    valid_ranks = t[tol .>= values]

    return valid_ranks

end

function set_filename(t_min::Int, condition_digit, condition_order)

    filename = "../coefficients_data/" 

    t_string = string(t_min)

    if length(t_string) == 1

        filename = filename * "1_xk0" 

    else 
        
        filename = filename * "1_xk" 

    end

    filename = filename * string(t_min) * "." * string(condition_digit) * "_" * string(condition_order)

    return filename

end

function obtain_coefficients(λ_min::T, κ::T, b_norm::T, tol::T) where T<:AbstractFloat 

    valid_ranks     = bound(λ_min, κ, b_norm, tol)
    println(valid_ranks)
    condition_order = Int(floor(log10(κ)))
    condition_digit = Int(floor(κ * inv(10^condition_order)))
    
    t_min = valid_ranks[1]

    # Construct file name
    filename = set_filename(t_min, condition_digit, condition_order)
    println(filename)
    println(isfile(filename))

    while (t_min <= valid_ranks[end]) && (!isfile(filename))

        t_min   += 1
        filename = set_filename(t_min, condition_digit, condition_order)

    end

    if t_min > valid_ranks[end]

        println("No valid rank was found")

        return
    end

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

function optimal_coefficients(df::DataFrame, τ::T, κ::T, λ::T, b̃_norm::T) where T<:AbstractFloat

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
    #mask      = γ .>= Vector(closest_row[1, :])
    mask = τ .>= Vector(closest_row[1, :])

    # Extract column headers (represent tensor ranks)
    matching_ranks = parse.(Int, names(closest_row[:, mask]) )

    # Take smallest rank that satisfies the bounds.
    t_min = minimum(matching_ranks)

    # Construct file name
    filename = "../coefficients_data/" 

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
