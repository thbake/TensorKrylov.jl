export ConvergenceData

mutable struct ConvergenceData{T} 

    niterations            ::Int
    iterations             ::Vector{Int}
    relative_residual_norm ::Vector{T}
    projected_residual_norm::Vector{T}
    orthogonality_data     ::Vector{T}

    function ConvergenceData{T}(nmax::Int) where T<:AbstractFloat

        new(
            nmax,
            collect(1:nmax),
            ones(nmax),
            ones(nmax),
            ones(nmax))

    end


end

function Base.resize!(convergencedata::ConvergenceData{T}, k::Int) where T<:AbstractFloat

    resize!(convergencedata.iterations, k)
    resize!(convergencedata.relative_residual_norm, k)
    resize!(convergencedata.projected_residual_norm, k)
    resize!(convergencedata.orthogonality_data, k)

end

function Base.show(io::IO, convergencedata::ConvergenceData{T}) where T<:AbstractFloat

    println(io, "Convergence data: ")
    println(io, " - Relative residual norm:  ", typeof(convergencedata.relative_residual_norm))
    println(io, " - Projected residual norm: ", typeof(convergencedata.projected_residual_norm))
    println(io, " - Orthogonality data:      ", typeof(convergencedata.orthogonality_data))

    println(io, "\nComputations ran for ",  convergencedata.niterations, " iterations.")
    println(io,
        "Achieved relative residual norm: ",
        convergencedata.relative_residual_norm[convergencedata.niterations]
    )
end
