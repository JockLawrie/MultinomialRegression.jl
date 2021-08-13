module MultinomialRegression

export fit, predict, L1, L2

include("regularization.jl")
include("fitpredict.jl")

using .regularization  # Independent
using .fitpredict      # Depends on: regularization

end
