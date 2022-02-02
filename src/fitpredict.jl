module fitpredict

export fit, predict, coef, stderror

using LinearAlgebra
using Logging
using Optim

using ..regularization

function fit(y, X, reg::Union{Nothing, AbstractRegularizer}=nothing, opts::Union{Nothing, Optim.Options}=nothing)
    # Fit
    nclasses = maximum(y)
    nx       = size(X, 2)
    probs    = fill(0.0, nclasses)
    B0       = fill(0.0, nx * (nclasses - 1))
    fg!      = get_fg!(reg, probs, y, X)  # Debug with opts = Optim.Options(show_trace=true)
    mdl      = isnothing(opts) ? optimize(Optim.only_fg!(fg!), B0, LBFGS()) : optimize(Optim.only_fg!(fg!), B0, LBFGS(), opts)
    theta    = mdl.minimizer
    coef     = reshape(theta, nx, nclasses - 1)
    reg isa BoxRegularizer && regularize!(coef, reg)  # Constrain coef to the box specified by reg

    # Collect stderror
    f      = TwiceDifferentiable(b -> loglikelihood_for_hessian!(y, X, b), theta; autodiff=:forward)
    hess   = Optim.hessian!(f, theta)
    ntheta = length(theta)
    if rank(hess) == ntheta
        varcov = inv(-hess)   # varcov(theta) = inv(FisherInformation) = inv(-Hessian)
        se     = [sqrt(abs(varcov[i,i])) for i = 1:ntheta]  # abs for negative values very close to 0
        se     = reshape(se, nx, nclasses - 1)
    else
        @warn "Hessian does not have full rank, therefore standard errors cannot be computed. Check for linearly dependent predictors."
        se = fill(NaN, nx, nclasses - 1)
    end
    (coef=coef, stderror=se)
end

predict(fittedmodel::NamedTuple, x) = predict(coef(fittedmodel), x)
predict(coef::Matrix, x)            = update_probs!(fill(0.0, 1 + size(coef, 2)), coef, x)

coef(fittedmodel)     = fittedmodel.coef
stderror(fittedmodel) = fittedmodel.stderror

################################################################################
# unexported

# L1 or L2 regularization
get_fg!(reg, probs, y, X) = (_, gradB, B) -> regularize(reg, B, gradB) - loglikelihood!(probs, y, X, B, gradB)

# No regularization
function get_fg!(reg::Nothing, probs, y, X)
    (_, gradB, B) -> begin
        fill!(gradB, 0.0)
        -loglikelihood!(probs, y, X, B, gradB)
    end
end

# Box regularization
function get_fg!(reg::BoxRegularizer, probs, y, X)
    nx       = size(X, 2)
    nclasses = length(probs)
    outB     = fill(0.0, nx * (nclasses - 1))
    (_, gradB, B) -> begin
        regularize!(outB, B, reg, gradB)
        -loglikelihood!(probs, y, X, outB, gradB)
    end
end

function loglikelihood!(probs, y, X, b::Vector, gradb::Vector)
    nx    = size(X, 2)
    nj    = Int(length(b) / nx)
    B     = reshape(b, nx, nj)
    gradB = reshape(gradb, nx, nj)
    loglikelihood!(probs, y, X, B, gradB)
end

function loglikelihood!(probs, y, X, B::Matrix, gradB::Matrix)
    LL = 0.0
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

"Called only for calculating the Hessian after optimization"
function loglikelihood_for_hessian!(y, X, b::AbstractVector{T}) where T
    nx    = size(X, 2)
    nj    = Int(length(b) / nx)
    B     = reshape(b, nx, nj)
    probs = zeros(T, 1 + nj)  # Must accommodate ForardDiff.Dual
    LL    = 0.0
    for (i, yi) in enumerate(y)
        x = view(X, i, :)
        update_probs!(probs, B, x)
        LL += log(max(probs[yi], 1e-12))
    end
    LL
end

end