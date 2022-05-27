module fitpredict

export MultinomialRegressionModel, @formula, fit, predict

using CategoricalArrays
using LinearAlgebra
using StatsModels

using ..regularization
using ..lbfgs

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
    xnames = c[2]
    X      = modelmatrix(f, data)  # Matrix{Float64}
    fit(y, X, yname, ylevels, xnames, wts, reg, solver, opts)
end

fit(y, X, wts=nothing, reg=nothing, solver=nothing, opts=nothing) = 
    fit(y, X, wts, sort!(unique(y)), ["x$(i)" for i = 1:size(X, 2)], reg, solver, opts)

predict(m::MultinomialRegressionModel, x) = _predict(m.coef, x)

################################################################################
# Unexported

_predict(b, x) = update_probs!(fill(0.0, 1 + size(b, 2)), b, x)

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

function update_probs!(probs, B, x)
    probs[1] = 0.0  # bx for first category is 0
    max_bx   = 0.0  # Find max bx for numerical stability
    nclasses = length(probs)
    for c = 2:nclasses
        bx       = dot(x, view(B, :, c - 1))
        probs[c] = bx
        max_bx   = bx > max_bx ? bx : max_bx
    end
    probs .= exp.(probs .- max_bx)  # Subtract max_bx first for numerical stability. Then prob[c] <= 1.
    normalize!(probs, 1)
end

end