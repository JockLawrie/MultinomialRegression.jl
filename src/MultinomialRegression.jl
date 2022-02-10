module MultinomialRegression

export MultinomialRegressionModel, @formula, fit, L1, L2, predict,  # Fit/predict
       nparams, coef, stderror, coeftable, coefcor, vcov,           # Coefficient diagnostics
       isregularized, nobs, loglikelihood, aic, aicc, bic           # Model diagnostics

# Fit/predict
include("regularization.jl")
include("fitpredict.jl")
using .regularization  # Independent
using .fitpredict      # Depends on: regularization

# Diagnostics
include("diagnostics.jl")
include("diagnostics_multinomial.jl")
using .diagnostics                # Independent
using .diagnostics_multinomial    # Depends on: diagnostics, fitpredict.MultinomialRegressionModel, deriveddiagnostics.jl

end
