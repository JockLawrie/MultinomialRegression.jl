module MultinomialRegression

export MultinomialRegressionModel, @formula, fit, L1, L2, predict,  # Construct and use model
       nparams, coef, stderror, coeftable, coefcor, vcov,           # Coefficient diagnostics
       isregularized, nobs, loglikelihood, aic, aicc, bic           # Model diagnostics

include("regularization.jl")
include("fitpredict.jl")

using .regularization  # Independent
using .fitpredict      # Depends on: regularization, ptables

end
