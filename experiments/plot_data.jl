using Plots
using LaTeXStrings

include("reproduction_spd.jl")

const ConvVec{T} = Vector{ConvergenceData{T}} 

abstract type AbstractDataPlot{T} end
abstract type CustomPlot{T} <: AbstractDataPlot{T} end

struct ResidualPlot{T} <: CustomPlot{T}

    data::Vector{Vector{T}}

    function ResidualPlot{T}(conv_data_vector::ConvVec{T}) where T<:AbstractFloat

        relative_residuals = [ conv_data_vector[i].relative_residual_norm for i in 1:length(conv_data_vector) ]

        new(relative_residuals)

    end

end

struct OrthogonalityPlot{T} <: CustomPlot{T}   

    data::Vector{Vector{T}}

    function OrthogonalityPlot{T}(conv_data_vector::ConvVec{T}) where T<:AbstractFloat

        orthogonality = [ conv_data_vector[i].orthogonality_data for i in 1:length(conv_data_vector) ]

        new(orthogonality)

    end

end

struct ProjResidualPlot{T} <: CustomPlot{T}

    data::Vector{Vector{T}}

    function ProjResidualPlot{T}(conv_data_vector::ConvVec{T}) where T<:AbstractFloat

        projected_residuals = [ conv_data_vector[i].projected_residual_norm for i in 1:length(conv_data_vector) ]

        new(projected_residuals)

    end

end

#struct SpectrumPlot{T} <: CustomPlot
#
#    conv_data_vector::ConvVec{T}
#
#    function SpectrumPlot{T}(conv_data_vector::ConvVec{T}) where T<:AbstractFloat
#
#        spectra_data = [ conv_data_vector.relative_residual_norm[i] for i in 1:length(conv_data_vector) ]
#
#        new(conv_data_vector)
#
#    end
#
#end

# Define Type Recipe for a vector of ConvergenceData{T}
@recipe function f(::Type{<:CustomPlot{T}}, custom_plot::CustomPlot{T}) where T<:AbstractFloat

    data = custom_plot.data

    return [ data[i] for i in 1:length(data)]

end

# Define Type Recipe for Experiment{T} 
@recipe function f(::Type{Experiment{T}}, experiment::Experiment{T}) where T<:AbstractFloat
    
    # type recipe for vector of ConvergenceData{T} is called recursively
    return experiment.conv_data_vector

end

# Define Plot Recipe for displaying experiments
@recipe function f(::Type{Val{:relativeresidual}}, plt::AbstractPlot) 

    x, y, z = plotattributes[:x], plotattributes[:y], plotattributes[:z]
    xlabel     --> L"k"
    ylabel     --> L"$\frac{||r_\mathfrak{K}||_2}{||b||_2}$"
    xlims      --> (1, 200)
    ylims      --> (1e-8, 1e+2)
    yscale     --> :log10
    yticks     --> 10.0 .^collect(-8:2:2)
    labels     --> permutedims(z)
    ls         --> :solid
    lw         --> 1.5
    marker     --> :circle
    markersize --> 1.5

    stride = 2:2:199

    x := x[stride]
    y := y[stride]
    seriestype := :path

end
@shorthands(relativeresidual)

@recipe function g(::Type{Val{:orthogonalityloss}}, plt::AbstractPlot)

    x, y, z = plotattributes[:x], plotattributes[:y], plotattributes[:z]
    xlabel     --> L"k"
    ylabel     --> L"$||\mathcal{V}_\mathfrak{K}^{H}\mathcal{V}_\mathfrak{K} - I_\mathfrak{K}||$"
    xlims      --> (1, 200)
    ylims      --> (1e-16, 1e+2)
    yscale     --> :log10
    yticks     --> 10.0 .^collect(-16:2:2)
    labels     --> permutedims(z)
    ls         --> :solid
    lw         --> 1.5
    marker     --> :circle
    markersize --> 1.5

    x := x
    y := y
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

    indices = (x .> 0) .& (y .> 0)
    x := x[indices]
    y := sqrt.(y[indices])
    seriestype := :path

end
@shorthands(proj)


function compute_labels(experiment::Experiment{T}) where T<:AbstractFloat

    return "d = " .* string.(experiment.dimensions)

end

function plot_experiment(experiment::Experiment{T}, ::Type{ResidualPlot{T}}) where T<:AbstractFloat

    x        = get_iterations(experiment)
    labels   = compute_labels(experiment)  # Add labels
    res_plot = ResidualPlot{T}(experiment.conv_data_vector)

    relativeresidual(x, res_plot, labels)

end


function plot_experiment(experiment::Experiment{T}, ::Type{OrthogonalityPlot{T}}) where T<:AbstractFloat

    x         = get_iterations(experiment)
    labels    = compute_labels(experiment)  # Add labels
    orth_plot = OrthogonalityPlot{T}(experiment.conv_data_vector)

    orthogonalityloss(x, orth_plot, labels)

end

function plot_experiment(experiment::Experiment{T}, ::Type{ProjResidualPlot{T}}) where T<:AbstractFloat

    x        = get_iterations(experiment)
    labels   = compute_labels(experiment)  # Add labels
    proj_res = ProjResidualPlot{T}(experiment.conv_data_vector)

    proj(x, proj_res, labels)

end
