module MLJinterface

export MultinomialLogisticRegressor

using Distributions
using MLJBase
using MLJModelInterface
using Tables

using ..regularization
using ..fitpredict
using ..diagnostics_multinomial

import ..fitpredict.fit  # To be overloaded

const MMI = MLJModelInterface

mutable struct MultinomialLogisticRegressor <: MLJBase.Probabilistic
    yname::String
    xnames::Union{Nothing, Vector{String}}
    wts::Union{Nothing, AbstractVector}
    solver::Union{Nothing, Symbol}
    reg::Union{Nothing, AbstractRegularizer}
    opts::Dict{Symbol, Any}
end

function MultinomialLogisticRegressor(; kwargs...)
    yname  = get(kwargs, :yname,  "y")
    xnames = get(kwargs, :xnames, nothing)
    wts    = get(kwargs, :wts,    nothing)
    solver = get(kwargs, :solver, nothing)
    reg    = get(kwargs, :reg,    nothing)
    opts   = get(kwargs, :opts,   Dict{Symbol, Any}())
    MultinomialLogisticRegressor(yname, xnames, wts, solver, reg, opts)
end

function MMI.fit(m::MultinomialLogisticRegressor, verbosity, X, y)
    Xmatrix   = MMI.matrix(X)
    sch       = MMI.schema(X)  # Table with colnames :names, :scitypes, :types
    xnames    = isnothing(sch) ? nothing : string.(sch.names)
    xnames    = isnothing(xnames) ? m.xnames : xnames
    fitresult = fitpredict.fit(y, Xmatrix; yname=m.yname, xnames=xnames, wts=m.wts, solver=m.solver, reg=m.reg, opts=m.opts)
    cache     = nothing
    report    = (stderror=stderror(fitted), vcov=vcov(fitted), loglikelihood=loglikelihood(fitted))
    return fitresult, cache, report
end

function MMI.predict(m::MultinomialLogisticRegressor, fitresult, Xnew)
    n      = MMI.nrows(Xnew)
    B      = fitresult.coef
    nx     = size(B, 1)
    x      = fill(0.0, nx)
    probs  = Vector{Vector{typeof(0.0)}}(undef, n)
    xnames = Symbol.(fitresult.xnames)
    rows   = Tables.rows(Xnew)
    for (i, row) in enumerate(rows)
        for (j, xnm) in enumerate(xnames)
            @inbounds x[j] = Tables.getcolumn(row, xnm)
        end
        @inbounds probs[i] = fitpredict._predict(B, x)
    end
    MMI.UnivariateFinite(fitresult.ylevels, probs)
end

end