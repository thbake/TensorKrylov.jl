function tensor_krylov(
        A::KronMat{T},
        b::KronProd{T},
        tol::T,
        nmax::Int,
        orthonormalization::Type{<:TensorDecomposition}) where T<:AbstractFloat

    d = length(A)

    tensor_decomposition = orthonormalization{T}(A)

    𝔎 = Vector{Int}(undef, d)

    b̃ = [ zeros( size(b[s]) ) for s in eachindex(b) ]

    x = zeros(nentries(A))

    for j = 2:nmax

        orthonormal_basis!(tensor_decomposition, b, j, Lanczos)

        update_rhs!(b̃, tensor_decomposition, b, j)

        H_minors = principal_minors(tensor_decomposition.H, j + 1)
        b_minors = principal_minors(b̃, j + 1)

        H_expanded = sparse(kroneckersum(H_minors.𝖳...))
        b_expanded = kronecker(b_minors...)

        prob = LinearProblem(H_expanded, b_expanded)

        linsolve = init(prob)

        solution = solve(linsolve, IterativeSolversJL_cg())


    end



end
