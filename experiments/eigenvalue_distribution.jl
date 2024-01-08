include("reproduction_spd.jl")

mutable struct EigenDist{T}

    dimensions::Vector{Int}
    niterations::Int
    conv_data_vector::Vector{ConvergenceData{T}}

end
