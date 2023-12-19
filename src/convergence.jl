export ConvergenceData

mutable struct ConvergenceData{T} 

    niterations            ::Int
    iterations             ::Vector{Int}
    relative_residual_norm ::Vector{T}
    projected_residual_norm::Vector{T}
    spectraldata           ::Vector{SpectralData{T}}
    orthogonality_data     ::Vector{T}

    function ConvergenceData{T}(nmax::Int) where T<:AbstractFloat

        spectraldata = [ SpectralData{T}() for _ in 1:nmax]

        new(
            nmax,
            collect(1:nmax),
            zeros(nmax),
            zeros(nmax),
            spectraldata,
            zeros(nmax))

    end

end

function Base.resize!(convergencedata::ConvergenceData{T}, k::Int) where T<:AbstractFloat

    resize!(convergencedata.iterations, k)
    resize!(convergencedata.relative_residual_norm, k)
    resize!(convergencedata.projected_residual_norm, k)
    resize!(convergencedata.spectraldata, k)
    resize!(convergencedata.orthogonality_data, k)

end
