module MultinomialRegression

export FittedMultinomialRegression, fit, predict, coef, stderror, L1, L2, BoxRegularizer, isregularized

include("regularization.jl")
include("fitpredict.jl")

using .regularization  # Independent
using .fitpredict      # Depends on: regularization

end
