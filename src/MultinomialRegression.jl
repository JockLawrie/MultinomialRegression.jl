module MultinomialRegression

export MultinomialRegressionModel, @formula, fit, predict,  # Fit/predict
       L1, L2, ElasticNet,                                  # Regularization
       nparams, coef, stderror, coeftable, coefcor, vcov,   # Coefficient diagnostics
       isregularized, nobs, loglikelihood, aic, aicc, bic   # Model diagnostics

# Fit/predict
include("regularization.jl")
#include("solvers/lbfgs.jl")
include("solvers/optim.jl")
include("fitpredict.jl")
using .regularization  # Independent
using .lbfgs           # Depends on: regularization
using .fitpredict      # Depends on: regularization

# Diagnostics
include("diagnostics.jl")
include("diagnostics_multinomial.jl")
using .diagnostics                # Independent
using .diagnostics_multinomial    # Depends on: fitpredict.MultinomialRegressionModel, diagnostics

end
