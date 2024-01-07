export ConvDiff, EigValMat, RandSPD, TensorizedSystem
export assemble_matrix, solve_tensorized_system

abstract type MatrixGallery end
struct Laplace   end <: MatrixGallery
struct ConvDiff  end <: MatrixGallery
struct EigValMat end <: MatrixGallery
struct RandSPD   end <: MatrixGallery

function assemble_matrix(n::Int, ::Laplace) where T<:AbstractFloat

    h  = inv(n + 1)
    Aₛ = inv(h^2) * sparse(SymTridiagonal(2ones(n), -ones(n)))

    return Aₛ

end

function assemble_matrix(n::Int, ::ConvDiff, c::AbstractFloat = 10.0) where T<:AbstractFloat

    h  = inv(n + 1)
    L  =     sparse( inv(h^2) .* SymTridiagonal( 2ones(n), -ones(n - 1)) )
    Aₛ = L + sparse( (c * inv(4 * h)) .* diagm(-1 => ones(n-1), 0 => 3ones(n), 1 => -5ones(n - 1), 2 => ones(n - 2)) )

    return Aₛ

end

function assemble_matrix(eigenvalues::Vector, ::EigenValMat)

    return diagm(eigenvalues)

end

function assemble_matrix(n::Int, ::RandSPD)

    R = rand(n, n)

    return  R'R

end

struct TensorizedSystem{T} 

    n                      ::Int
    d                      ::Int
    A                      ::KronMat{T}
    b                      ::KronProd{T}
    orthonormalization_type::Type{<:TensorDecomposition{T}}

    function TensorizedSystem{T}(
        n                     ::Int,
        d                     ::Int,
        matrix                ::MatrixGallery,
        orthogonalization_type::Type{<:TensorDecomposition{T}},
        normalize_rhs         ::Bool = true) where T

        Aₛ = assemble_matrix(n, MatrixGallery())
        bₛ = rand(n)
        A  = KronMat{T}([Aₛ for _ in 1:d])
        b  = [ bₛ for _ in 1:d ]
        
        if normalize_rhs

            normalize!(b)

        end

        new(n, d, A, b, orthogonalization_type)

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

function Base.show(io::IO, system::TensorizedSystem) 

    display(system)

end

function solve_tensorized_system(system::TensorizedSystem{T}, nmax::Int, tol::T = 1e-9) where T

    convergencedata = ConvergenceData{Float64}(nmax)

    tensor_krylov!(
        convergencedata, system.A,
        system.b,
        tol,
        nmax,
        system.orthonormalization_type
    )

    return convergencedata

end
