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

function Base.show(io::IO, convergencedata::ConvergenceData{T}) where T<:AbstractFloat

    println("Convergence data: Computations ran for ",  convergencedata.niterations)
    println(" - Relative residual norm:  ", typeof(convergencedata.relative_residual_norm))
    println(" - Projected residual norm: ", typeof(convergencedata.projected_residual_norm))
    println(" - Spectral data:           ", typeof(convergencedata.spectraldata))
    println(" - Orthogonality data:      ", typeof(convergencedata.orthogonality_data))

end

#function process_convergence!(
#    convergencedata   ::ConvergenceData{T},
#    k                 ::Int,
#    boolean_dictionary::Dic{AbstractString, Bool}) where T<:AbstractFloat
#
#    
#
#
#
#end
