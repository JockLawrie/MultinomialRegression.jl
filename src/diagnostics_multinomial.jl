"""
Implementation of the diagnostics API for instances of MultinomialRegressionModel.

These functions are defined here:
- coef(m), vcov(m), stderror(m), nobs(m), nparams(m), loglikelihood(m), isregularized(m)

These functions are automatically derived:
- aic(m), aicc(m), bic(m), coefcor(m), coeftable(m, confidence_level)
"""
module diagnostics_multinomial

export isregularized, nobs, loglikelihood,  # Model diagnostics
       nparams, coef, stderror, vcov,       # Coefficient diagnostics
       aic, aicc, bic, coefcor, coeftable   # Diagnostics derived from other diagnostics

import ..fitpredict: MultinomialRegressionModel
using ..diagnostics
import ..diagnostics: coef, vcov, stderror, nobs, nparams, loglikelihood, isregularized  # To be overloaded

# Model-level diagnostics (aic, aicc, bic, coefcor, coeftable are defined in the diagnostics module)
isregularized(m::MultinomialRegressionModel) = _isregularized(m.loglikelihood, m.loss)
nobs(m::MultinomialRegressionModel)          = m.nobs
loglikelihood(m::MultinomialRegressionModel) = m.loglikelihood

# Coefficient-level diagnostics
nparams(m::MultinomialRegressionModel)  = _nparams(m.coef)
coef(m::MultinomialRegressionModel)     = _coef(m.coef, m.xnames, m.ylevels[2:end])
stderror(m::MultinomialRegressionModel) = _stderror(m.vcov, m.xnames, m.ylevels[2:end])
vcov(m::MultinomialRegressionModel)     = _vcov(m.vcov, m.xnames, m.ylevels[2:end])

end