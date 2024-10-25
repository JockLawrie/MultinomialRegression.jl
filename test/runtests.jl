using Test
using MultinomialRegression

using CategoricalArrays
using DataFrames
using Dates
using Logging
using RDatasets
using Statistics

@testset "ReadMe" begin

@info "$(now()) Starting test set: ReadMe"
iris  = dataset("datasets", "iris")
model = fit(@formula(Species ~ 1 + SepalWidth), iris)  # levels(iris.Species)[1] is the reference category
opts  = Dict(:iterations => 250, :g_abstol => 1e-8)    # Same terminology as Optim.Options
model = fit(@formula(Species ~ 1 + SepalWidth), iris; opts=opts)
Btrue = [18.85843663920683 12.997324428062807; -6.118961548679346 -4.0790981064343335]
@test isapprox(coef(model), Btrue; atol=1e-8)

end

include("iris_binary.jl")
include("iris.jl")
include("alligators.jl")

@info "$(now()) Finished"