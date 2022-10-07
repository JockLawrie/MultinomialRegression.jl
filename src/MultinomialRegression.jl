module MultinomialRegression

export MultinomialRegressionModel, @formula, fit, predict, predict!,  # Fit/predict
       L1, L2, ElasticNet,                                  # Regularization
       nparams, coef, stderror, coeftable, coefcor, vcov,   # Coefficient diagnostics
       isregularized, nobs, loglikelihood, aic, aicc, bic,  # Model diagnostics
       MultinomialLogisticRegressor                         # MLJ interface

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

# MLJ interface
include("MLJinterface.jl")
using .MLJinterface

end
