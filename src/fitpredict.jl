module fitpredict

export MultinomialRegressionModel, fit, predict, L1, L2,   # Construct and use model
       nparams, coef, stderror, coeftable, coefcor, vcov,  # Coefficient diagnostics
       isregularized, nobs, loglikelihood, aic, aicc, bic  # Model diagnostics

using AxisArrays
using Distributions
using LinearAlgebra
using Logging
using Optim
using PrettyTables
using StatsBase
import Base: show  # Overload for 1D and 2D AxisArrays

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

"""
Assumptions:
1. y has categories numbered 1 to nclasses.
2. The first category is the reference category.
"""
function fit(y, X, yname::String, ylevels::Vector{String}, xnames::Vector{String},
             reg::Union{Nothing, AbstractRegularizer}=nothing, opts::Union{Nothing, Optim.Options}=nothing)
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
    nparams = length(theta)
    loss    = mdl.minimum
    ylvls   = length(ylevels) == nclasses - 1 ? ylevels : ylevels[2:end]
    if isnothing(reg)  # Unregularized model => coef is the maximum-likelihood estimate
        LL   = -loss
        f    = TwiceDifferentiable(b -> loglikelihood(y, X, b), theta; autodiff=:forward)
        hess = Optim.hessian!(f, theta)
        if rank(hess) == nparams
            vcov = inv(-hess)   # varcov(theta) = inv(FisherInformation) = inv(-Hessian)
        else
            @warn "Hessian does not have full rank, therefore standard errors cannot be computed. Check for linearly dependent predictors."
            vcov = fill(NaN, nparams, nparams)
        end
    else
        @warn "The parameter covariance matrix and standard errors are not available for regularized regression."
        vcov = fill(NaN, nparams, nparams)
        LL   = penalty(reg, theta) - loss  # loss = -LL + penalty
    end
    MultinomialRegressionModel(yname, ylvls, xnames, coef, vcov, nobs, LL, loss)
end

function fit(y, X, reg::Union{Nothing, AbstractRegularizer}=nothing, opts::Union{Nothing, Optim.Options}=nothing)
    yname   = "y"
    ylevels = ["y$(i)" for i = 1:maximum(y)]
    xnames  = ["x$(i)" for i = 1:size(X, 2)]
    fit(y, X, yname, ylevels, xnames, reg, opts)
end

predict(m::MultinomialRegressionModel, x) = predict(m.coef, x)
predict(b::Matrix, x) = update_probs!(fill(0.0, 1 + size(b, 2)), b, x)

nparams(m::MultinomialRegressionModel)       = length(m.coef)
nobs(m::MultinomialRegressionModel)          = m.nobs
loglikelihood(m::MultinomialRegressionModel) = m.loglikelihood
isregularized(m::MultinomialRegressionModel) = m.loss != -m.loglikelihood  # If not regularized, then loss == -LL

coef(m::MultinomialRegressionModel) = AxisArray(m.coef, rownames=m.xnames, colnames=m.ylevels)

function stderror(m::MultinomialRegressionModel)
    se   = [sqrt(abs(m.vcov[i,i])) for i = 1:nparams(m)]
    data = reshape(se, size(m.coef))
    AxisArray(data, rownames=m.xnames, colnames=m.ylevels)
end

function coeftable(m::MultinomialRegressionModel, ci_level::Float64=0.95)
    levstr   = isinteger(ci_level*100) ? string(Integer(ci_level*100)) : string(ci_level*100)
    ni       = nparams(m)
    b        = reshape(m.coef, ni)
    se       = [sqrt(abs(m.vcov[i,i])) for i = 1:ni]
    z        = b ./ se
    pvals    = 2 * ccdf.(Ref(Normal()), abs.(z))
    ci_width = -se * quantile(Normal(), (1-ci_level)/2)
    data     = hcat(b, se, z, pvals, b-ci_width, b+ci_width)
    colnames = ["Coef", "StdError", "z", "Pr(>|z|)", "Lower $levstr%", "Upper $levstr%"]
    rownames = construct_vcov_rownames(m, true)
    AxisArray(data, rownames=rownames, colnames=colnames)
end

function vcov(m::MultinomialRegressionModel)
    colnames = construct_vcov_rownames(m, false)
    rownames = construct_vcov_rownames(m, true)
    AxisArray(m.vcov, rownames=rownames, colnames=colnames)
end

function coefcor(m::MultinomialRegressionModel)
    se       = [sqrt(abs(m.vcov[i,i])) for i = 1:nparams(m)]
    data     = StatsBase.cov2cor(m.vcov, se)
    colnames = construct_vcov_rownames(m, false)
    rownames = construct_vcov_rownames(m, true)
    AxisArray(data, rownames=rownames, colnames=colnames)
end

function aic(m::MultinomialRegressionModel)
    isregularized(m) && @warn "Model is regularized. AIC is not based on MLE."
    2.0 * (nparams(m) - loglikelihood(m))
end

function aicc(m::MultinomialRegressionModel)
    LL = loglikelihood(m)
    k  = nparams(m)
    n  = nobs(m)
    isregularized(m) && @warn "Model is regularized. AICc is not based on MLE."
    2.0*(k - LL) + 2.0*k*(k - 1.0)/(n - k - 1.0)
end

function bic(m::MultinomialRegressionModel)
    isregularized(m) && @warn "Model is regularized. BIC is not based on MLE."
    -2.0*loglikelihood(m) + nparams(m)*log(nobs(m))
end

################################################################################
# Unexported

function construct_vcov_rownames(m::MultinomialRegressionModel, align_vertically::Bool)
    if align_vertically  # Suitable for row names
        ylevel_maxlen  = maximum(length.(m.ylevels)) + 4  # +2 for "y=_  "
        ylevels_padded = rpad.(["y=$(ylevel)  " for ylevel in m.ylevels], ylevel_maxlen)
        xname_maxlen   = maximum(length.(m.xnames)) + 2   # +2 for "x="
        xnames_padded  = rpad.(["x=$(xname)" for xname in m.xnames], xname_maxlen)
        reshape(["$(ylevel)$(xname)" for xname in xnames_padded, ylevel in ylevels_padded], nparams(m))
    else                 # Suitable for column names
        reshape(["y=$(ylevel)  x=$(xname)" for xname in m.xnames, ylevel in m.ylevels], nparams(m))
    end
end

function Base.show(io::IO, ::MIME"text/plain", table::AxisArray{T,1,D,Tuple{A}}) where {T,D,A}
    pretty_table(table.data, header=[""], row_names=collect(table.axes[1].val))
end

function Base.show(io::IO, ::MIME"text/plain", table::AxisArray{T,2,D,Tuple{A1,A2}}) where {T,D,A1,A2}
    pretty_table(table.data, header=collect(table.axes[2].val), row_names=collect(table.axes[1].val))
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