module fitpredict

export FittedMultinomialRegression, fit, predict, coef, stderror, isregularized

using LinearAlgebra
using Logging
using Optim

using ..regularization

struct FittedMultinomialRegression
    coef::Matrix{Float64}
    vcov::Matrix{Float64}
    stderror::Matrix{Float64}
    nobs::Int
    dof::Int  # The number of fitted parameters (the dimension of the optimization problem)
    loglikelihood::Float64
    loss::Float64
end

"""
Assumptions:
1. y has categories numbered 1 to nclasses.
2. The reference category is the category numbered 1.
"""
function fit(y, X, reg::Union{Nothing, AbstractRegularizer}=nothing, opts::Union{Nothing, Optim.Options}=nothing)
    # Fit
    nclasses = maximum(y)
    nx       = size(X, 2)
    probs    = fill(0.0, nclasses)
    B0       = fill(0.0, nx * (nclasses - 1))
    fg!      = get_fg!(reg, probs, y, X)
    mdl      = isnothing(opts) ? optimize(Optim.only_fg!(fg!), B0, LBFGS()) : optimize(Optim.only_fg!(fg!), B0, LBFGS(), opts)
    theta    = mdl.minimizer

    # Construct result
    coef    = reshape(theta, nx, nclasses - 1)
    nobs    = length(y)
    dof     = length(theta)
    loss    = mdl.minimum
    if isnothing(reg)  # Unregularized model => coef is the maximum-likelihood estimate
        LL   = -loss
        f    = TwiceDifferentiable(b -> loglikelihood(y, X, b), theta; autodiff=:forward)
        hess = Optim.hessian!(f, theta)
        if rank(hess) == dof
            vcov = inv(-hess)   # varcov(theta) = inv(FisherInformation) = inv(-Hessian)
            se   = [sqrt(abs(vcov[i,i])) for i = 1:dof]  # abs for negative values very close to 0
            se   = reshape(se, nx, nclasses - 1)
        else
            @warn "Hessian does not have full rank, therefore standard errors cannot be computed. Check for linearly dependent predictors."
            vcov = fill(NaN, dof, dof)
            se   = fill(NaN, nx, nclasses - 1)
        end
    else
        @warn "The parameter covariance matrix and standard errors are not available for regularized regression."
        vcov = fill(NaN, dof, dof)
        se   = fill(NaN, nx, nclasses - 1)
        LL   = penalty(reg, theta) - loss  # loss = -LL + penalty
    end
    FittedMultinomialRegression(coef, vcov, se, nobs, dof, LL, loss)
end

predict(fitted::FittedMultinomialRegression, x) = predict(coef(fitted), x)
predict(coef::Matrix, x) = update_probs!(fill(0.0, 1 + size(coef, 2)), coef, x)

coef(fitted::FittedMultinomialRegression)     = fitted.coef
dof(fitted::FittedMultinomialRegression)      = fitted.dof
stderror(fitted::FittedMultinomialRegression) = fitted.stderror
vcov(fitted::FittedMultinomialRegression)     = fitted.vcov
nobs(fitted::FittedMultinomialRegression)     = fitted.nobs
loglikelihood(fitted::FittedMultinomialRegression) = fitted.loglikelihood
isregularized(fitted::FittedMultinomialRegression) = fitted.loss != -fitted.loglikelihood  # If not regularized, then loss == -LL

function aic(fitted::FittedMultinomialRegression)
    isregularized(fitted) && @warn "Model is regularized. AIC is not based on MLE."
    2.0 * (dof(fitted) - loglikelihood(fitted))
end

function aicc(fitted::FittedMultinomialRegression)
    LL = loglikelihood(fitted)
    k  = dof(fitted)
    n  = nobs(fitted)
    isregularized(fitted) && @warn "Model is regularized. AICc is not based on MLE."
    2.0*(k - LL) + 2.0*k*(k - 1.0)/(n - k - 1.0)
end

function bic(fitted::FittedMultinomialRegression)
    isregularized(fitted) && @warn "Model is regularized. BIC is not based on MLE."
    -2.0*loglikelihood(fitted) + dof(fitted)*log(nobs(fitted))
end

################################################################################
# unexported

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