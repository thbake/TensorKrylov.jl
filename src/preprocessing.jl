using DataFrames
using CSV

export extract_coefficients

function extract_coefficients(τ::T, κ::T, λ::T, b̃_norm::T) where T<:AbstractFloat

    # Extracts coefficients αⱼ, ωⱼ > 0 and tensor rank t such that the bound of 
    # Lemma 2.6 (denoted by γ) is satisfied.
    #
    # This has been already computed for multiple pairs of 
    # t (tensor rank), and R = κ. 
    #
    # In order to satisfy the bound of Corollary 2.7, (γ / λ) ⋅ ||b̃||₂ ≤ τ, where
    # τ is the desired tolerance. This is equivalent as looking for the value
    #
    #   γ ≤ (τ ⋅ λ) / ||b̃||₂
    #

    data = "../coefficients_data/output_data/k_R_accuracy_processed.csv"

    column_types = [ fill(Float64, 64)... ]

    # Compute desired tolerance
    γ = λ * inv(b̃_norm) * τ

    # Compute order
    condition_order = Int(floor(log10(κ)))

    first_digit = Int( floor(κ / (10^condition_order)) )

    df = CSV.read(data, DataFrame, delim = ',', types = column_types)

    # Find row that matches best the condition number 
    closest_row = filter(row -> row.R ≈ first_digit * 10^condition_order, df)[:, 2:end]

    # Take ranks whose corresponding accuracy is below γ
    mask = γ .<= Vector(closest_row[1, :])

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
                    delim = "   ",
                    header = ["number", "id"]
                )


    ω = parse.(Float64, coeffs_df[1:t_min, 1])
    α = parse.(Float64, coeffs_df[t_min + 1 : 2t_min, 1])

    return ω, α, t_min

end
