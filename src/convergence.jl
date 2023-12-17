export ConvergenceData

mutable struct ConvergenceData{T} 

    niterations::Int
    iterations::Vector{Int}
    relative_residual_norm::Vector{T}
    projected_residual_norm::Vector{T}
    

    function ConvergenceData{T}(nmax::Int) where T<:AbstractFloat

        new(nmax, collect(1:nmax), zeros(nmax), zeros(nmax))

    end

end
