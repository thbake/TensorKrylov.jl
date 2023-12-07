export ConvergenceData

struct ConvergenceData{T} 

    iterations::Vector{Int}
    relative_residualnorm::Vector{T}

    function ConvergenceData{T}(nmax::Int) where T<:AbstractFloat

        new(collect(1:nmax), zeros(nmax))

    end

end


