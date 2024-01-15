export compute_eigdist, plot_dist_comparison, plot_laplace

compute_eigdist(D::EigValDist) = [ possiblesums(D.eigenvalues, d) for d ∈ D.dims ]

compute_eigdist(dims::Vector{Int}, vals::Vector) = [ possiblesums(vals, d) for d ∈ dims ]

function plot_eigdist(distributions, labels) 

    α = 0.5
    rev_dist = reverse(distributions)
    rev_lab  = permutedims(reverse(labels))

    histogram(rev_dist, label = rev_lab, fillalpha = α)
    xlabel!(L"$\mathbb{R}$")

end

function plot_clusterone(dims = [3, 5, 7], n = 15)

    goodeigs = clusterone(n)

    eigenkrongood = compute_eigdist(dims, goodeigs)
    labels        = compute_labels(dims)

    plot_eigdist(eigenkrongood, labels)

end

function plot_clusterzero(dims = [3, 5, 7], n = 15)

    badeigs = clusterzero(n)

    eigenkronbad = compute_eigdist(dims, badeigs)
    labels       = compute_labels(dims)

    plot_eigdist(eigenkronbad, labels)

end

function plot_dist_comparison(dims = [3, 5, 7], n = 15)

    plt1 = plot_clusterone(dims, n)
    plt2 = plot_clusterzero(dims, n)

    plot(plt1, plt2, layout = (1,2))

    

end

function plot_laplace(dims = [5, 7, 9], n = 15)

    L      = assemble_matrix(n, Laplace{Float64})
    values = eigvals(Matrix(L))

    distributions = compute_eigdist(dims, values)
    labels        = compute_labels(dims)

    plot_eigdist(distributions, labels)

end

