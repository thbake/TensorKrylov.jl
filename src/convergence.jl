export ConvergenceData

mutable struct ConvergenceData{T} 

    niterations            ::Int
    iterations             ::Vector{Int}
    relative_residual_norm ::Vector{T}
    projected_residual_norm::Vector{T}
    spectraldata           ::SpectralData{T}
    orthogonality_data     ::Vector{T}

    function ConvergenceData{T}(nmax::Int, instance::Type{<:Instance}) where T<:AbstractFloat

        new(
            nmax,
            collect(1:nmax),
            ones(nmax),
            ones(nmax),
            SpectralData{T, instance}(nmax),
            ones(nmax))

    end


end

function Base.resize!(convergencedata::ConvergenceData{T}, k::Int) where T<:AbstractFloat

    resize!(convergencedata.iterations, k)
    resize!(convergencedata.relative_residual_norm, k)
    resize!(convergencedata.projected_residual_norm, k)
    resize!(convergencedata.spectraldata, k)
    resize!(convergencedata.orthogonality_data, k)

end

function Base.show(::IO, convergencedata::ConvergenceData{T}) where T<:AbstractFloat

    println("Convergence data: ")
    println(" - Relative residual norm:  ", typeof(convergencedata.relative_residual_norm))
    println(" - Projected residual norm: ", typeof(convergencedata.projected_residual_norm))
    println(" - Spectral data:           ", typeof(convergencedata.spectraldata))
    println(" - Orthogonality data:      ", typeof(convergencedata.orthogonality_data))

    println("\nComputations ran for ",  convergencedata.niterations, " iterations.")
    println(
        "Achieved relative residual norm: ",
        convergencedata.relative_residual_norm[convergencedata.niterations]
    )
end
