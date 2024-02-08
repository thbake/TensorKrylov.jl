export compute_eigdist, plot_dist_comparison, plot_laplace

compute_eigdist(D::EigValDist, T::Type{<:EVC}) = [ possiblesums(D.eigenvalues, d, T()) for d ∈ D.dims ]

compute_eigdist(dims::Vector{Int}, vals::Vector, T::Type{<:EVC}) = [ possiblesums(vals, d, T()) for d ∈ dims ]

compute_mean(eigenvalues::Vector{T}) where T = inv(length(eigenvalues)) * sum(eigenvalues)

function plot_eigdist(distributions, labels) 

    rev_dist = reverse(distributions)
    rev_lab  = permutedims(reverse(labels))

    histogram(rev_dist, label = rev_lab)
    xlabel!(L"$\mathbb{R}$")

end

function plot_clusterone(T::Type{<:EVC}, dims = [3, 5, 7], n = 15)

    goodeigs = clusterone(n)

    eigenkrongood = compute_eigdist(dims, goodeigs, T)
    labels        = compute_labels(dims)

    plot_eigdist(eigenkrongood, labels)

end

function plot_clusterzero(T::Type{<:EVC}, dims = [3, 5, 7], n = 15)

    badeigs = clusterzero(n)

    eigenkronbad = compute_eigdist(dims, badeigs, T)
    labels       = compute_labels(dims)

    plot_eigdist(eigenkronbad, labels)

end

function plot_dist_comparison(T::Type{<:EVC}, dims = [3, 5, 7], n = 15)

    plt1 = plot_clusterzero(T, dims, n)
    plt2 = plot_clusterone(T,  dims, n)

    plot(plt1, plt2, layout = (1,2))

    

end

function plot_laplace(T::Type{<:EVC}, dims = [5, 7, 9], n = 15)

    L      = assemble_matrix(n, Laplace)
    values = eigvals(Matrix(L))

    distributions = compute_eigdist(dims, values, T)
    labels        = compute_labels(dims)

    plot_eigdist(distributions, labels)

end

