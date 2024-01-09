export TensorizedSystem
export multiple_rhs, random_rhs, solve_tensorized_system


function random_rhs(d::Int, n::Int)

    bs = rand(n)

    return [ bs for _ in 1:d ]

end

multiple_rhs(dims::Array{Int}, n::Int) = [ random_rhs(d, n) for d ∈ dims ]

struct TensorizedSystem{T} 

    d                      ::Int
    n                      ::Int
    A                      ::KronMat{T}
    b                      ::KronProd{T}
    orthonormalization_type::Type{<:TensorDecomposition{T}}

    function TensorizedSystem{T}(
        A                     ::KronMat{T},
        b                     ::KronProd{T},
        orthogonalization_type::Type{<:TensorDecomposition{T}},
        normalize_rhs         ::Bool = true) where T

        @assert length(A) == length(b)
        @assert all(dimensions(A) .== size.(b, 1))

        d = length(A)
        n = size(first(A), 1)

        if normalize_rhs

            normalize!(b)

        end


        new(d, n, A, b, orthogonalization_type)

    end

end

function display(system::TensorizedSystem) 

    println(
        "Tensorized linear system of order d = ",
        system.d,
        "  with coefficient matrices of order n = ",
        system.n
    )
        flush(stdout)

end

function Base.show(::IO, system::TensorizedSystem) 

    display(system)

end

function solve_tensorized_system(system::TensorizedSystem{T}, nmax::Int, tol::T = 1e-9) where T

    convergencedata = ConvergenceData{T}(nmax)

    tensor_krylov!(
        convergencedata, system.A,
        system.b,
        tol,
        nmax,
        system.orthonormalization_type,
    )

    return convergencedata

end
