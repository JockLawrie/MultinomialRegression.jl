#=
Requires type-specific methods for the following functions:
- coef(m), vcov(m), nobs(m), nparams(m), loglikelihood(m), isregularized(m)
=#

using AxisArrays
using Distributions
using StatsBase

function aic(m)
    isregularized(m) && @warn "Model is regularized. AIC is not based on MLE."
    2.0 * (nparams(m) - loglikelihood(m))
end

function aicc(m)
    LL = loglikelihood(m)
    k  = nparams(m)
    n  = nobs(m)
    isregularized(m) && @warn "Model is regularized. AICc is not based on MLE."
    2.0*(k - LL) + 2.0*k*(k - 1.0)/(n - k - 1.0)
end

function bic(m)
    isregularized(m) && @warn "Model is regularized. BIC is not based on MLE."
    -2.0*loglikelihood(m) + nparams(m)*log(nobs(m))
end

function coefcor(m)
    v        = vcov(m)
    ni       = size(v, 1)
    se       = [sqrt(abs(v[i,i])) for i = 1:ni]
    data     = StatsBase.cov2cor(v, se)
    colnames = collect(v.axes[2].val)
    rownames = collect(v.axes[1].val)
    AxisArray(data, rownames=rownames, colnames=colnames)
end

function coeftable(m, ci_level::Float64=0.95)
    b        = coef(m)
    v        = vcov(m)
    levstr   = isinteger(ci_level*100) ? string(Integer(ci_level*100)) : string(ci_level*100)
    ni       = length(b)
    bvec     = reshape(b, ni)
    se       = [sqrt(abs(v[i,i])) for i = 1:ni]
    z        = bvec ./ se
    pvals    = 2 * ccdf.(Ref(Normal()), abs.(z))
    ci_width = -se * quantile(Normal(), (1-ci_level)/2)
    data     = hcat(bvec, se, z, pvals, bvec-ci_width, bvec+ci_width)
    colnames = ["Coef", "StdError", "z", "Pr(>|z|)", "Lower $levstr%", "Upper $levstr%"]
    rownames = collect(v.axes[1].val)
    AxisArray(data, rownames=rownames, colnames=colnames)
end