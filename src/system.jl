export TensorizedSystem
export multiple_rhs, random_rhs, solve_tensorized_system


function random_rhs(d::Int, n::Int)

    bs = rand(n)

    return [ bs for _ in 1:d ]

end

multiple_rhs(dims::Array{Int}, n::Int) = [ random_rhs(d, n) for d ∈ dims ]

struct TensorizedSystem{U} 

    d                      ::Int
    n                      ::Int
    A                      ::KronMat
    b                      ::KronProd

    function TensorizedSystem{U}(
        A                     ::KronMat{matT, U},
        b                     ::KronProd,
        normalize_rhs         ::Bool = true) where {matT, U<:Instance}

        @assert length(A)          == length(b)
        @assert all(dimensions(A) .== size.(b, 1))

        d = length(A)
        n = size(first(A), 1)

        if normalize_rhs

            normalize!(b)

        end

        new{U}(d, n, A, b)

    end

end

getinstancetype(::TensorizedSystem{U}) where {U} = U

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

function solve_tensorized_system(
    system                ::TensorizedSystem,
    nmax                  ::Int,
    orthogonalization_type::Type{<:TensorDecomposition},
    tol                   ::T = 1e-9) where {T}

    convergencedata = ConvergenceData{T}(nmax)

    tensorkrylov!(
        convergencedata, system.A,
        system.b,
        tol,
        nmax,
        orthogonalization_type,
    )

    return convergencedata

end
