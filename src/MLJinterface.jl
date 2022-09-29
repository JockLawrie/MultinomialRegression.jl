module MLJinterface

export MultinomialLogisticRegressor

using Distributions
using MLJBase
using MLJModelInterface
using Tables

using ..fitpredict
using ..diagnostics_multinomial

const MMI = MLJModelInterface

mutable struct MultinomialLogisticRegressor <: MLJBase.Probabilistic
end

function MMI.fit(model::MultinomialLogisticRegressor, verbosity, X, y)
    ylevels   = MMI.classes(y[1])  # ylevels as a CategoricalVector
    length(ylevels) < 2 && throw(DomainError("The response variable must have two or more levels."))
    yint      = MMI.int(y)
    Xmatrix   = MMI.matrix(X)
    sch       = MMI.schema(d)  # Table with colnames :names, :scitypes, :types
    xnames    = isnothing(sch) ? nothing : [string(nm) for nm in sch.names]
    model     = fitpredict.fit(yint, Xmatrix, "y", levels(ylevels), xnames)
    fitresult = (model=model, ylevels=ylevels)
    cache     = nothing
    report    = (stderror=stderror(model), vcov=vcov(model), loglikelihood=loglikelihood(model))
    return fitresult, cache, report
end

function MMI.predict(m::MultinomialLogisticRegressor, fitresult, Xnew)
    n      = MMI.nrows(Xnew)
    model  = fitresult.model
    B      = model.coef
    nx     = size(B, 1)
    x      = fill(0.0, nx)
    probs  = Vector{Vector{typeof(0.0)}}(undef, n)
    xnames = Symbol.(model.xnames)
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