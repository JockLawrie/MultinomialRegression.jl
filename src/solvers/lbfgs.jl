module lbfgs

export fit_lbfgs

using LinearAlgebra
using Logging
using Optim

using ..regularization

function fit_lbfgs(y, X, wts::Union{Nothing, AbstractVector}=nothing,
                   reg::Union{Nothing, AbstractRegularizer}=nothing, opts::Union{Nothing, Optim.Options}=nothing)
    # Fit
    nclasses = length(unique(y))
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
    nobs    = length(y)
    probs   = fill(0.0, nobs, nclasses)
    for i = 1:nobs
        update_probs!(view(probs, i, :), coef, view(X, i, :))
    end
    H = hessian(X, probs, wts, reg)
    if rank(H) == nparams
        if penalty(reg, theta) == 0.0
            @warn "Regularisation implies that the covariance matrix and standard errors are not estimated from MLEs"
        end
        vcov = Matrix(Hermitian(inv(bunchkaufman!(-H))))  # varcov(theta) = inv(FisherInformation) = inv(-Hessian)
    else
        @warn "Standard errors cannot be computed (Hessian does not have full rank). Check for linearly dependent predictors."
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

getweight(wts::Nothing, i) = nothing
getweight(wts, i)          = wts[i]

LL_delta(w::Nothing, probs, yi) = log(max(probs[yi], 1e-12))
LL_delta(w, probs, yi)          = log(max(probs[yi], 1e-12)) * w

weighted_grad(w::Nothing, p, xj) = p * xj
weighted_grad(w, p, xj)          = p * xj * w

weighted_grad(w::Nothing, xj) = xj
weighted_grad(w, xj)          = xj * w

"""
Let k = the number of categories, and let p = the number of predictors.
The hessian is a (k-1) x (k-1) block matrix, with block size p x p.
In the code below, i and j denote the block indices; i.e., i and j each have k-1 values.
"""
@views function hessian!(H, X, probs, wts, reg)
    k  = size(probs, 2)  # nclasses
    p  = size(X, 2)      # npredictors
    Xt = transpose(X)
    for j = 1:(k - 1)
        for i = j:(k - 1)
            rows = (p*(i - 1) + 1):(p*i)
            cols = (p*(j - 1) + 1):(p*j)
            if i == j
                H[rows, cols] = Xt * Diagonal(probs[:, i + 1] .* (probs[:, i + 1] .- 1) .* wts) * X
            else
                H[rows, cols] = Xt * Diagonal(probs[:, i + 1] .* probs[:, j + 1] .* wts) * X
            end
        end
    end
    H = Hermitian(H, :L)
    penalty_hessian!(H, reg)
    H
end

function hessian(X, probs, wts, reg)
    k  = size(probs, 2)  # nclasses
    p  = size(X, 2)      # npredictors
    H  = fill(0.0, p*(k - 1), p*(k - 1))
    hessian!(H, X, probs, wts, reg)
end

hessian(X, probs, wts::Nothing, reg) = hessian(X, probs, fill(1.0, size(X, 1)), reg)

end