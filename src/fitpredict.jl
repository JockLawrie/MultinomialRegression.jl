module fitpredict

export MultinomialRegressionModel, @formula, fit, predict, predict!

using CategoricalArrays
using LinearAlgebra
using StatsModels
using Tables

using ..regularization
using ..optim

const FLOAT = typeof(0.0)

struct MultinomialRegressionModel{T}
    yname::String
    ylevels::Vector{T}
    xnames::Vector{String}
    coef::Matrix{FLOAT}
    vcov::Matrix{FLOAT}
    nobs::Int
    loglikelihood::FLOAT
    loss::FLOAT
end

"""
Assumptions:
1. The first category of the response variable is the reference category.
2. X is a Matrix{typeof(0.0)}.
"""
function fit(y::AbstractVector, X::AbstractMatrix{<: Real};
             yname::String="y", xnames::Union{Nothing, Vector{String}}=nothing,
             wts::Union{Nothing, AbstractVector}=nothing,
             reg::Union{Nothing, AbstractRegularizer}=nothing, opts::Union{Nothing, AbstractDict}=nothing)
    y, ylevels = construct_y_and_ylevels(y)
    length(ylevels) < 2 && throw(DomainError("The response variable must have at least two levels."))
    xnames = isnothing(xnames) ? ["x$(i)" for i = 1:size(X, 2)] : xnames
    format_weights!(wts)
    loss, coef, vcov = fit_optim(y, X, wts, reg, opts)
    nobs  = length(y)
    LL    = penalty(reg, reshape(coef, length(coef))) - loss  # loss = -LL + penalty
    MultinomialRegressionModel(yname, ylevels, xnames, coef, vcov, nobs, LL, loss)
end

function fit(f::FormulaTerm, data; kwargs...)
    yname  = string(f.lhs.sym)
    f      = apply_schema(f, schema(f, data))
    c      = coefnames(f)  # (names(y), names(x))
    xnames = c[2] isa String ? [c[2]] : c[2]
    kwargs = update_kwargs(kwargs, :yname => yname, :xnames => xnames)
    y = data[!, yname]
    X = modelmatrix(f, data)  # Matrix{Float64}
    fit(y, X; kwargs...)
end

function predict!(probs, B::Matrix, x::AbstractVector)
    probs[1] = 0.0
    probs[2:end] .= reshape(x' * B, size(B, 2))
    optim.softmax!(probs)
    probs
end

predict!(probs, m::MultinomialRegressionModel, x::AbstractVector) = predict!(probs, m.coef, x)

function predict!(probs, m::MultinomialRegressionModel, row)
    x = [Tables.getcolumn(row, Symbol(xnm)) for xnm in m.xnames]
    predict!(probs, m, x)
end

function predict!(probs, m::MultinomialRegressionModel, row, xworkspace)
    for (j, xnm) in enumerate(m.xnames)
        @inbounds xworkspace[j] = Tables.getcolumn(row, Symbol(xnm))
    end
    predict!(probs, m, xworkspace)
end

function predict(m::MultinomialRegressionModel, x::AbstractVector)
    B = m.coef
    probs = fill(0.0, 1 + size(B, 2)) # nclasses = 1 + size(B, 2)
    predict!(probs, B, x)
end

function predict(m::MultinomialRegressionModel, row)
    x = [Tables.getcolumn(row, Symbol(xnm)) for xnm in m.xnames]
    predict(m, x)
end

################################################################################
# Unexported

update_kwargs(kwargs::Nothing, ps...) = Dict(p[1] => p[2] for p in ps)

function update_kwargs(kwargs, ps...)
    result = Dict{Symbol, Any}()
    for (k, v) in kwargs
        result[k] = v
    end
    for p in ps
        result[p[1]] = p[2]
    end
    result
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