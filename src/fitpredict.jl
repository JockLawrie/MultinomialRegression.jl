module fitpredict

export MultinomialRegressionModel, @formula, fit, L1, L2, predict

using CategoricalArrays
using LinearAlgebra
using Logging
using Optim
using StatsModels
using ..regularization

struct MultinomialRegressionModel
    yname::String
    ylevels::Vector{String}
    xnames::Vector{String}
    coef::Matrix{Float64}
    vcov::Matrix{Float64}
    nobs::Int
    loglikelihood::Float64
    loss::Float64
end

function fit(f::FormulaTerm, data, reg::Union{Nothing, AbstractRegularizer}=nothing, opts::Union{Nothing, Optim.Options}=nothing)
    yname  = string(f.lhs.sym)
    y, ylevels = construct_y_and_ylevels(data[!, yname])
    f      = apply_schema(f, schema(f, data))
    c      = coefnames(f)  # (names(y), names(x))
    xnames = c[2]
    X      = modelmatrix(f, data)  # Matrix{Float64}
    fit(y, X, yname, ylevels, xnames, reg, opts)
end

function fit(y, X, yname, ylevels::Nothing, xnames, reg, opts)
    y, ylevels = construct_y_and_ylevels(data[!, yname])
    fit(ycat, X, yname, ylevels, xnames, reg, opts)
end

"""
Assumptions:
1. y has categories numbered 1 to nclasses.
2. The first category is the reference category.
"""
function fit(y, X, yname::String, ylevels::Vector{String}, xnames::Vector{String},
             reg::Union{Nothing, AbstractRegularizer}=nothing, opts::Union{Nothing, Optim.Options}=nothing)
    # Fit
    nclasses = length(ylevels)
    nx       = size(X, 2)
    probs    = fill(0.0, nclasses)
    B0       = fill(0.0, nx * (nclasses - 1))
    fg!      = get_fg!(reg, probs, y, X)
    mdl      = isnothing(opts) ? optimize(Optim.only_fg!(fg!), B0, LBFGS()) : optimize(Optim.only_fg!(fg!), B0, LBFGS(), opts)
    theta    = mdl.minimizer

    # Construct result
    coef    = reshape(theta, nx, nclasses - 1)
    nobs    = length(y)
    loss    = mdl.minimum
    LL      = isnothing(reg) ? -loss : penalty(reg, theta) - loss  # loss = -LL + penalty
    nparams = length(theta)
    if isapprox(-LL, loss, atol=1e-8)  # Estimated parameters are MLEs or very close to
        f    = TwiceDifferentiable(b -> loglikelihood(y, X, b), theta; autodiff=:forward)
        hess = Optim.hessian!(f, theta)
        if rank(hess) == nparams
            vcov = inv(-hess)   # varcov(theta) = inv(FisherInformation) = inv(-Hessian)
        else
            @warn "Standard errors cannot be computed (Hessian does not have full rank). Check for linearly dependent predictors."
            vcov = fill(NaN, nparams, nparams)
        end
    else
        @warn "The covariance matrix and standard errors are not available because the parameters are not maximum likelihood estimates"
        vcov = fill(NaN, nparams, nparams)
    end
    MultinomialRegressionModel(yname, ylevels, xnames, coef, vcov, nobs, LL, loss)
end

predict(m::MultinomialRegressionModel, x) = _predict(m.coef, x)
_predict(b, x) = update_probs!(fill(0.0, 1 + size(b, 2)), b, x)

################################################################################
# Unexported functions for fitting a multinomial regression model

"Construct categorical y and ylevels when the supplied y is not categorical."
construct_y_and_ylevels(ydata::CategoricalVector) = ydata.refs, levels(ydata)

function construct_y_and_ylevels(ydata::AbstractVector)
    yvals   = sort!(unique(ydata))
    ycatvec = categorical(ydata; levels=yvals)
    ylevels = string.(levels(ycatvec))
    y       = ycatvec.refs
    y, ylevels
end

# No regularization
get_fg!(reg::Nothing, probs, y, X) = (_, gradb, b) -> -loglikelihood!(probs, gradb, y, X, b)

# L1 or L2 regularization
function get_fg!(reg, probs, y, X)
    (_, gradb, b) -> begin
        loss = -loglikelihood!(probs, gradb, y, X, b) + penalty(reg, b)
        penalty_gradient!(gradb, reg, b)
        loss
    end
end

function loglikelihood(y, X, b::AbstractVector{T}) where T
    nx       = size(X, 2)
    nclasses = Int(length(b) / nx) + 1
    probs    = zeros(T, nclasses)             # Accommodates ForwardDiff.Dual
    gradb    = zeros(T, nx * (nclasses - 1))  # Accommodates ForwardDiff.Dual
    loglikelihood!(probs, gradb, y, X, b)
end

function loglikelihood!(probs, gradb, y, X, b)
    fill!(gradb, 0.0)
    LL    = 0.0
    nx    = size(X, 2)
    nj    = Int(length(b) / nx)  # nclasses - 1
    B     = reshape(b, nx, nj)
    gradB = reshape(gradb, nx, nj)
    for (i, yi) in enumerate(y)
        x = view(X, i, :)
        update_probs!(probs, B, x)
        update_gradient!(gradB, probs, yi, x)
        LL += log(max(probs[yi], 1e-12))
    end
    LL
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

function update_gradient!(gradB, probs, yi, x)
    nclasses = length(probs)
    nx = length(x)
    for c = 2:nclasses
        p = probs[c]
        for (j, xj) in enumerate(x)
            gradB[j, c - 1] += p * xj  # Negative gradient because function is to be minimized
        end
    end
    yi == 1 && return nothing
    for (j, xj) in enumerate(x)
        gradB[j, yi - 1] -= xj  # Negative gradient because function is to be minimized
    end
    nothing
end

end