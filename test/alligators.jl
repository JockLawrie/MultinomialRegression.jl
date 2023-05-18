@testset "Alligators" begin

@info "$(now()) Starting test set: Alligators"

#=
  Stomach contents of alligators in different lakes.
  https://bookdown.org/mpfoley1973/statistics/generalized-linear-models-glm.html#multinomial-logistic-regression
=#
include("setup_alligators.jl")

# Train
model = fit(y, X; yname=yname, xnames=xnames)

# Assess fitted parameters and standard errors
target_params   = transpose(reshape(target[:, 2], 4, 6))
fitted_params   = coef(model)
target_stderror = transpose(reshape(target[:, 3], 4, 6))
fitted_stderror = stderror(model)

@test maximum(abs.(fitted_params .- target_params)) <= 0.0001
@test maximum(abs.(fitted_stderror .- target_stderror)) <= 0.0001

end