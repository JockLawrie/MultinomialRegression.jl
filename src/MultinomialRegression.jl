module MultinomialRegression

export fit, predict

include("regularization.jl")
include("fitpredict.jl")

using .regularization  # Independent
using .fitpredict      # Depends on: regularization

end
