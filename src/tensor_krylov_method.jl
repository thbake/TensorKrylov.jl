export tensor_krylov, update_rhs! 

using ExponentialUtilities: exponential!, expv
using SparseArrays: mul!



function matrix_exponential_vector!(
        y::ktensor,
        A::KronMat{T},
        b::KronProd{T},
        γ::T,
        k::Int) where T<:AbstractFloat

    for s = 1:length(A)

        tmp = Matrix(copy(A[s]))

        y.fmat[s][:, k] = expv(γ, tmp, b[s])

    end

end

function solve_compressed_system(
        H::KronMat{T}, 
        b::Vector{<:AbstractVector{T}}, 
        ω::Array{T},
        α::Array{T},
        t::Int,
        λ::T,
    ) where T <: AbstractFloat

    reciprocal = inv(λ)

    # Since we are considering a canonical decomposition the tensor rank of yₜ
    # is equal to 

    k = dimensions(H)
    
    yₜ = ktensor(reciprocal .* ω, [ ones(k[s], t) for s in 1:length(H)] )

    for k = 1:t

        γ = -α[k] * reciprocal

        matrix_exponential_vector!(yₜ, H, b, γ, k)

    end

    return yₜ
end



function initialize_compressed_rhs(b::KronProd{T}, V::KronMat{T}) where T<:AbstractFloat

        b̃        = [ zeros( size(b[s]) )  for s in eachindex(b) ]
        b_minors = principal_minors(b̃, 1)
        columns  = kth_columns(V, 1)
        update_rhs!(b_minors, columns, b, 1)

        return b̃
end

function update_rhs!(b̃::KronProd{T}, V::KronProd{T}, b::KronProd{T}, k::Int) where T<:AbstractFloat
    # b̃ = Vᵀb = ⨂ Vₛᵀ ⋅ ⨂ bₛ = ⨂ Vₛᵀbₛ
    
    for s = 1:length(b̃)

        # Update one entry of each component of b̃ by performing a single inner product 
        b̃[s][k] = dot(V[s], b[s])

    end

end

function basis_tensor_mul!(x::ktensor, V::KronMat{T}, y::ktensor) where T<:AbstractFloat

    x.lambda = copy(y.lambda)

    for s in eachindex(V)

        LinearAlgebra.mul!(x.fmat[s], V[s], y.fmat[s])

    end

end


function initialize!(
        A::KronMat{T},
        b::KronProd{T},
        b̃::KronProd{T},
        t_orthonormalization::Type{<:TensorDecomposition}) where T <: AbstractFloat


    # Initialize the d Arnoldi decompositions of Aₛ
    tensor_decomposition = t_orthonormalization(A)

    orthonormal_basis!(tensor_decomposition, b, 1, tensor_decomposition.orthonormalization)

    for s in 1:length(A)

        b̃[s][1] = prod(tensor_decomposition.V[1, 1]) * b[s][1]

    end

    return tensor_decomposition

end


# SPD case
function tensor_krylov(
        A::KronMat{T},
        b::KronProd{T},
        tol::T,
        nmax::Int,
        t_orthonormalization::Type{<:TensorDecomposition}) where T <: AbstractFloat

	# Initilialize implicit tensorized Krylov subspace basis and upper Hessenberg 
    d = length(A)

    # Initialize multiindex 𝔎
    𝔎 = Vector{Int}(undef, d)


    # Allocate memory for approximate solution
    x = nothing

    tensor_decomp = t_orthonormalization(A)

    # Initialize list of characteristic polynomials of Jacobi matrices Tₖ
    char_poly = CharacteristicPolynomials{T}(d, tensor_decomp.H[1, 1])

    orthonormalization = tensor_decomp.orthonormalization
    initial_orthonormalization!(tensor_decomp, b, orthonormalization)

    # Allocate memory for right-hand side b̃
    b̃ = initialize_compressed_rhs(b, tensor_decomp.V)

    #omega = [
    #    0.0000001270914523635023453823152008025368,   
    #    0.0000008331874358562597752320087564582775,   
    #    0.0000034080672022105782788001866918936589,   
    #    0.0000110739933654662785807222409336476269,   
    #    0.0000311667995071938174038291518550035722,   
    #    0.0000793037753836722995497952373169462531,   
    #    0.0001870512355599002836650045670054999525,   
    #    0.0004156014090353547126400450014262547005,   
    #    0.0008795380745097793068189862279900037567,   
    #    0.0017872115492601083069274024803935185801,   
    #    0.0035079904326112304643424925519254453654,   
    #    0.0066823414684595417136606424445000040890,   
    #    0.0123992217962518246942221289766394853871,   
    #    0.0224779652765556349558256302861858344500,   
    #    0.0399106362824236598495805487807341904727,   
    #    0.0695487353045896257146426221174007054060,   
    #    0.1191608188830656863852960891669852117047,   
    #    0.2010592077384481716302230824844343715085,   
    #    0.3346412689951259481163734627040540203780,   
    #    0.5505930392537740739348703367106452333246,   
    #    0.8989022904885930039193056573232354367065,   
    #    1.4686339627311775085649614425165054854006,   
    #    2.4576919868563932352820144977556537924102,   
    #    4.5959866102993954576315382976048340424313,   
    #]

    #alpha = [

    #    0.0000000397972590548516429192696767192358,
    #    0.0000004367491128819478938719430765502788,
    #    0.0000023090182209582046634003901011960181,
    #    0.0000089066906550732628370716519314645703,
    #    0.0000285117175356736196462963597113272028,
    #    0.0000804187756982368456452923229007406780,
    #    0.0002066731175280465565669122562269047205,
    #    0.0004942168102835682589995968023739020270,
    #    0.0011153306788409162644606638196964620846,
    #    0.0023994711022991415183188839170200234996,
    #    0.0049578632395921716233139491143699917330,
    #    0.0098951272990664324780065320280586504964,
    #    0.0191621474132992261945940054221337867091,
    #    0.0361348098865640136563917583778504649672,
    #    0.0665492040412706367295088069613306913652,
    #    0.1199928740446123454391983559341738896364,
    #    0.2122553292353875994974031146678150605567,
    #    0.3689998821842512652011657203754424472208,
    #    0.6314804412918880311398651949961191576222,
    #    1.0655364534650591431521812757488021361496,
    #    1.7763959759375297358264395031213211950671,
    #    2.9361676873642319511234499707086342823459,
    #    4.8496969121493506472454770861446604612865,
    #    8.2047528053762217824240732255702823749743,
    #]

    if t_orthonormalization == TensorLanczos{T}

        coefficients_df = compute_dataframe()

    end

    n = dimensions(A)[1]

    for k = 2:nmax

        # Compute orthonormal basis and Hessenberg factor of each Krylov subspace 𝓚ₖ(Aₛ, bₛ) 
        orthonormal_basis!(tensor_decomp, k)

        H_minors = principal_minors(tensor_decomp.H, k)
        V_minors = principal_minors(tensor_decomp.V, n, k)
        b_minors = principal_minors(b̃, k)

        #λ_min, λ_max = extreme_tensorized_eigenvalues(H_minors, char_poly, k)
        λ_min, λ_max = tensor_qr_algorithm(H_minors, 1e-5, 100)

        columns = kth_columns(tensor_decomp.V, k)

        # Update compressed right-hand side b̃ = Vᵀb
        update_rhs!(b_minors, columns, b, k)

        b_norm = kronprodnorm(b_minors)

        κ = abs(λ_max / λ_min)

        if κ < 1

            κ = 2.0

        end

        @info "Condition: " κ
        #@info "Smallest eigenvalue:" λ_min 
        #@info "b_norm: " b_norm

        ω, α, rank = optimal_coefficients_mod(coefficients_df, tol, κ, λ_min, b_norm)
        
        #rank = 24
        
        @info "Chosen tensor rank: " rank

        # Approximate solution of compressed system
        #y = solve_compressed_system(H_minors, b_minors, omega, alpha, rank, λ_min)
        y = solve_compressed_system(H_minors, b_minors, ω, α, rank, λ_min)

        𝔎 .= k 

        subdiagonal_entries = [ tensor_decomp.H[s][k + 1, k] for s in 1:d ]

        # Compute residual norm
        r_norm = residual_norm(H_minors, y, 𝔎, subdiagonal_entries, b_minors)

        rel_res_norm = (r_norm / kronprodnorm(b_minors))

        @info "Iteration: " k "relative residual norm:" rel_res_norm


        if rel_res_norm < tol

            x = ktensor( ones(rank), [ zeros(size(A[s], 1), rank) for s in 1:d ])

            x_minors = principal_minors(x, k)

            basis_tensor_mul!(x_minors, V_minors, y)

            println("Convergence")

            return x

        end

    end

    println("No convergence")

end

function tensor_krylov(
        A::KronMat{T},
        b::KronProd{T},
        tol::T,
        nmax::Int,
        t_orthonormalization::Type{TensorArnoldi{T}}) where T <: AbstractFloat

	# Initilialize implicit tensorized Krylov subspace basis and upper Hessenberg 
    d = length(A)

    # Initialize multiindex 𝔎
    𝔎 = Vector{Int}(undef, d)

    # Allocate memory for right-hand side b̃
    b̃ = [ zeros( size(b[s]) )  for s in eachindex(b) ]

    # Allocate memory for approximate solution
    x = nothing

    t_arnoldi = t_orthonormalization(A)

    initial_orthonormalization!(t_arnoldi, b, Arnoldi)

    for k = 2:nmax

        # Compute orthonormal basis and Hessenberg factor of each Krylov subspace 𝓚ₖ(Aₛ, bₛ) 
        orthonormal_basis!(t_arnoldi, k)

        H_minors = principal_minors(t_arnoldi.H, k)
        V_minors = principal_minors(t_arnoldi.V, k)
        b_minors = principal_minors(b̃, k)

        #λ_min, λ_max = extreme_tensorized_eigenvalues(H_minors, char_poly, k)
        λ_min, λ_max = tensor_qr_algorithm(H_minors, 1e-5, 100)
        

        @info "Eigenvalues" λ_min, λ_max

        columns = kth_columns(t_arnoldi.V, k)

        # Update compressed right-hand side b̃ = Vᵀb
        update_rhs!(b_minors, columns, b)

        b_norm = kronprodnorm(b_minors)

        κ = abs(λ_max / λ_min)

        @info "Condition: " κ
        #@info "Smallest eigenvalue:" λ_min 
        #@info "b_norm: " b_norm

        ω, α, rank = optimal_coefficients_mod(coefficients_df, tol, κ, λ_min, b_norm)

        @info "Chosen tensor rank: " rank

        # Approximate solution of compressed system
        y = solve_compressed_system(H_minors, b_minors, ω, α, rank, λ_min)

        𝔎 .= k 

        subdiagonal_entries = [ tensor_decomp.H[s][k + 1, k] for s in 1:d ]

        # Compute residual norm
        r_norm = residual_norm(H_minors, y, 𝔎, subdiagonal_entries, b_minors)

        rel_res_norm = (r_norm / kronprodnorm(b_minors))

        @info "Iteration: " k "relative residual norm:" rel_res_norm


        if rel_res_norm < tol

            x = ktensor( ones(rank), [ zeros(size(A[s], 1), rank) for s in 1:d ])

            x_minors = principal_minors(x, k)

            basis_tensor_mul!(x_minors, V_minors, y)

            println("Convergence")

            return x

        end

    end

    println("No convergence")

    return x

end
