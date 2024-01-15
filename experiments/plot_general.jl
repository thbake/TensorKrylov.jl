export OrthogonalityPlot, ProjResidualPlot, ResidualPlot
export plot_experiment

using Plots

function getstride(v::Vector{T}, s::Int) where T

    return 1:s:length(v) 

end

# Define Type Recipe for Reproduction{T} 
@recipe function f(::Type{<:Experiment{T}}, experiment::Experiment{T}) where T
    
    # type recipe for vector of ConvergenceData{T} is called recursively
    return experiment.conv_vector

end


# Define Plot Recipe for displaying experiments
@recipe function f(::Type{Val{:relativeresidual}}, plt::AbstractPlot; n::Int, point_sep = 1) # Here n is a kw-arg
    x, y, z = plotattributes[:x], plotattributes[:y], plotattributes[:z]
    xlabel     --> L"k"
    ylabel     --> L"$\frac{||r_\mathfrak{K}||_2}{||b||_2}$"
    xlims      --> (1, n)
    #ylims      --> (1e-8, 1e+2)
    yscale     --> :log10
    #yticks     --> 10.0 .^collect(-8:2:2)
    labels     --> permutedims(z)
    ls         --> :solid
    lw         --> 1.5
    marker     --> :circle
    markersize --> 1.5

    point_sep_array = getstride(x, point_sep)
    x := x[point_sep_array]
    y := y[point_sep_array]

    seriestype := :path
end
@shorthands(relativeresidual)

@recipe function g(::Type{Val{:orthogonalityloss}}, plt::AbstractPlot)

    x, y, z = plotattributes[:x], plotattributes[:y], plotattributes[:z]
    xlabel     --> L"k"
    ylabel     --> L"$||\mathcal{V}_\mathfrak{K}^{H}\mathcal{V}_\mathfrak{K} - I_\mathfrak{K}||$"
    xlims      --> (1, length(x))
    ylims      --> (1e-16, 1e+2)
    yscale     --> :log10
    yticks     --> 10.0 .^collect(-16:2:2)
    labels     --> permutedims(z)
    ls         --> :solid
    lw         --> 1.5
    marker     --> :circle
    markersize --> 1.5

    stride = 2:5:199

    x := x[stride]
    y := y[stride]
    seriestype := :path
end
@shorthands(orthogonalityloss)

@recipe function f(::Type{Val{:proj}}, plt::AbstractPlot) 

    x, y, z = plotattributes[:x], plotattributes[:y], plotattributes[:z]
    xlabel     --> L"k"
    ylabel     --> L"$||\mathcal{H}_\mathfrak{K} y_t - \tilde{b}||_2$"
    xlims      --> (1, 200)
    xticks     --> collect(0:50:200)
    #ylims      --> (1e-8, 1e+2)
    yscale     --> :log10
    yticks     --> 10.0 .^collect(-100:10:2)
    labels     --> permutedims(z)
    ls         --> :solid
    lw         --> 1.5
    marker     --> :circle
    markersize --> 1.5

    stride = 2:2:199
    indices = (x .> 0) .& (y .> 0)
    #x := x[indices]
    #y := sqrt.(y[indices])
    x := x[stride]
    y := sqrt.(y[stride])
    seriestype := :path

end
@shorthands(proj)

abstract type CustomPlot end

# Define Type Recipe for a vector of ConvergenceData{T}
@recipe function f(::Type{<:CustomPlot}, custom_plot::CustomPlot) 

    data = custom_plot.data

    return [ data[i] for i in 1:length(data)]

end

struct ResidualPlot <: CustomPlot

    data  ::Vector{Vector}
    series::Symbol 

    function ResidualPlot(conv_vector::ConvVec{T}) where T<:AbstractFloat

        relative_residuals = [ conv_vector[i].relative_residual_norm for i in 1:length(conv_vector) ]

        new(relative_residuals, :relativeresidual)

    end

end

struct OrthogonalityPlot <: CustomPlot   

    data  ::Vector{Vector}
    series::Symbol 

    function OrthogonalityPlot(conv_vector::ConvVec{T}) where T<:AbstractFloat

        orthogonality = [ conv_vector[i].orthogonality_data for i in 1:length(conv_vector) ]

        new(orthogonality, :orthogonalityloss)

    end

end

struct ProjResidualPlot <: CustomPlot

    data  ::Vector{Vector}
    series::Symbol

    function ProjResidualPlot(conv_vector::ConvVec{T}) where T<:AbstractFloat

        projected_residuals = [ conv_vector[i].projected_residual_norm for i in 1:length(conv_vector) ]

        new(projected_residuals, :proj)

    end

end


function plot_experiment(
    experiment ::Experiment,
    custom_plot::Type{<:CustomPlot}, 
    point_sep::Int = 1) 

    x        = get_iterations(experiment)
    labels   = compute_labels(experiment)  # Add labels
    res_plot = custom_plot(experiment.conv_vector)
    

    relativeresidual(x, res_plot, labels, n = experiment.matrix_size, point_sep = point_sep)

end
