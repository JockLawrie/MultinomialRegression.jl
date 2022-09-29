module fitpredict

export MultinomialRegressionModel, @formula, fit, predict

using CategoricalArrays
using LinearAlgebra
using StatsModels
using Tables

using ..regularization
using ..optim

struct MultinomialRegressionModel{T}
    yname::String
    ylevels::Vector{T}
    xnames::Vector{String}
    coef::Matrix{Float64}
    vcov::Matrix{Float64}
    nobs::Int
    loglikelihood::Float64
    loss::Float64
end

"""
Assumptions:
1. y has categories numbered 1 to nclasses.
2. The first category is the reference category.
"""
function fit(y, X, yname::String, ylevels::AbstractVector, xnames::Vector{String}, wts::Union{Nothing, AbstractVector}=nothing,
             reg::Union{Nothing, AbstractRegularizer}=nothing, solver=nothing, opts::Union{Nothing, T}=nothing) where{T}
    format_weights!(wts)
    loss, coef, vcov = fit_optim(y, X, wts, reg, solver, opts)
    nobs = length(y)
    LL   = penalty(reg, reshape(coef, length(coef))) - loss  # loss = -LL + penalty
    MultinomialRegressionModel(yname, ylevels, xnames, coef, vcov, nobs, LL, loss)
end

function fit(f::FormulaTerm, data; wts::Union{Nothing, AbstractVector}=nothing,
             reg::Union{Nothing, AbstractRegularizer}=nothing, solver=nothing, opts::Union{Nothing, T}=nothing) where{T}
    yname  = string(f.lhs.sym)
    y, ylevels = construct_y_and_ylevels(data[!, yname])
    f      = apply_schema(f, schema(f, data))
    c      = coefnames(f)  # (names(y), names(x))
    xnames = c[2] isa String ? [c[2]] : c[2]
    X      = modelmatrix(f, data)  # Matrix{Float64}
    fit(y, X, yname, ylevels, xnames, wts, reg, solver, opts)
end

fit(y, X, wts=nothing, reg=nothing, solver=nothing, opts=nothing) = 
    fit(y, X, sort!(unique(y)), ["x$(i)" for i = 1:size(X, 2)], wts, reg, solver, opts)

predict(m::MultinomialRegressionModel, x::AbstractVector) = _predict(m.coef, x)

function predict(m::MultinomialRegressionModel, row)
    x = [Tables.getcolumn(row, Symbol(xnm)) for xnm in m.xnames]
    predict(m, x)
end

################################################################################
# Unexported

function _predict(B, x)
    nclasses = 1 + size(B, 2)
    probs    = fill(0.0, nclasses)
    probs[2:end] .= reshape(x' * B, nclasses - 1)
    optim.softmax!(probs)
    probs
end
    
"Construct categorical y and ylevels when the supplied y is not categorical."
construct_y_and_ylevels(ydata::CategoricalVector) = ydata.refs, levels(ydata)

function construct_y_and_ylevels(ydata::AbstractVector)
    yvals   = sort!(unique(ydata))
    ycatvec = categorical(ydata; levels=yvals)
    ycatvec.refs, levels(ycatvec)
end

"""
Ensure that:
- No weights are missing.
- No weights are negative.
- The weights sum to the number of observations.
"""
function format_weights!(wts)
    isnothing(wts) && return nothing
    s = 0.0
    for w in wts
        ismissing(w) && error("Weights cannot be missing.")
        w < 0.0 && error("Weights must be non-negative.")
        s += w
    end
    n = length(wts)
    s == n && return nothing
    m = n / s
    wts .*= m
    nothing
end

end