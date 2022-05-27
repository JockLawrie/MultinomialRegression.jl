module lbfgs

export fit_lbfgs

using LinearAlgebra
using Logging
using Optim

using ..regularization

function fit_lbfgs(y, X, ylevels::AbstractVector, wts::Union{Nothing, AbstractVector}=nothing,
                   reg::Union{Nothing, AbstractRegularizer}=nothing, opts::Union{Nothing, Optim.Options}=nothing)
    # Fit
    nclasses = length(ylevels)
    nx       = size(X, 2)
    probs    = fill(0.0, nclasses)
    B0       = fill(0.0, nx * (nclasses - 1))
    fg!      = get_fg!(reg, probs, y, X, wts)
    mdl      = isnothing(opts) ? optimize(Optim.only_fg!(fg!), B0, LBFGS()) : optimize(Optim.only_fg!(fg!), B0, LBFGS(), opts)
    theta    = mdl.minimizer

    # Construct result
    coef    = reshape(theta, nx, nclasses - 1)
    loss    = mdl.minimum
    LL      = penalty(reg, theta) - loss  # loss = -LL + penalty
    nparams = length(theta)
    if isapprox(-LL, loss, atol=1e-8)  # Estimated parameters are MLEs or very close to
        f    = TwiceDifferentiable(b -> loglikelihood(y, X, b, wts), theta; autodiff=:forward)
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
    loss, coef, vcov
end

################################################################################
# Unexported

function get_fg!(reg, probs, y, X, wts)
    (_, gradb, b) -> begin
        loss = -loglikelihood!(probs, gradb, y, X, b, wts) + penalty(reg, b)
        penalty_gradient!(gradb, reg, b)
        loss
    end
end

# This method is called when computing standard errors
function loglikelihood(y, X, b::AbstractVector{T}, wts) where T
    nx       = size(X, 2)
    nclasses = Int(length(b) / nx) + 1
    probs    = zeros(T, nclasses)             # Accommodates ForwardDiff.Dual
    gradb    = zeros(T, nx * (nclasses - 1))  # Accommodates ForwardDiff.Dual
    loglikelihood!(probs, gradb, y, X, b, wts)
end

getweight(wts::Nothing, i) = nothing
getweight(wts, i)          = wts[i]

LL_delta(w::Nothing, probs, yi) = log(max(probs[yi], 1e-12))
LL_delta(w, probs, yi)          = log(max(probs[yi], 1e-12)) * w

function loglikelihood!(probs, gradb, y, X, b, wts)
    fill!(gradb, 0.0)
    LL    = 0.0
    nx    = size(X, 2)
    nj    = Int(length(b) / nx)  # nclasses - 1
    B     = reshape(b, nx, nj)
    gradB = reshape(gradb, nx, nj)
    for (i, yi) in enumerate(y)
        w = getweight(wts, i)
        x = view(X, i, :)
        update_probs!(probs, B, x)
        update_gradient!(gradB, probs, yi, x, w)
        LL += LL_delta(w, probs, yi)
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

function update_gradient!(gradB, probs, yi, x, w)
    nclasses = length(probs)
    nx = length(x)
    for c = 2:nclasses
        p = probs[c]
        for (j, xj) in enumerate(x)
            gradB[j, c - 1] += weighted_grad(w, p, xj)  # Negative gradient because function is to be minimized
        end
    end
    yi == 1 && return nothing
    for (j, xj) in enumerate(x)
        gradB[j, yi - 1] -= weighted_grad(w, xj)  # Negative gradient because function is to be minimized
    end
    nothing
end

weighted_grad(w::Nothing, p, xj) = p * xj
weighted_grad(w, p, xj)          = p * xj * w

weighted_grad(w::Nothing, xj) = xj
weighted_grad(w, xj)          = xj * w

end