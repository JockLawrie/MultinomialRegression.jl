module MultinomialRegression

export MultinomialRegressionModel, @formula, fit, predict, predict!,  # Fit/predict
       L1, L2, ElasticNet,                                 # Regularization
       nparams, coef, stderror, coeftable, coefcor, vcov,  # Coefficient diagnostics
       isregularized, nobs, loglikelihood, aic, aicc, bic  # Model diagnostics

# Fit/predict
include("regularization.jl")
include("optim.jl")
include("fitpredict.jl")
using .regularization  # Independent
using .optim           # Depends on: regularization
using .fitpredict      # Depends on: regularization

# Diagnostics
include("diagnostics.jl")
include("diagnostics_multinomial.jl")
using .diagnostics                # Independent
using .diagnostics_multinomial    # Depends on: fitpredict.MultinomialRegressionModel, diagnostics

# Pre-compilation
using PrecompileTools

@setup_workload begin
    using CategoricalArrays
    include("../test/setup_alligators.jl")
    w = fill(1.0, size(X, 1))
    xnew = X[1, :]

    @compile_workload begin
       model = fit(y, X; yname=yname, xnames=xnames)
       m2    = fit(y, X; yname=yname, xnames=xnames, wts=w)
       ynew  = predict(model, xnew)
       isregularized(model)
       nobs(model)
       loglikelihood(model)
       aic(model)
       aicc(model)
       bic(model)
       nparams(model)
       coef(model)
       coeftable(model)
       stderror(model)
       vcov(model)
       coefcor(model)
    end
end

end
