using Test
using MultinomialRegression

using DataFrames
using Dates
using Logging
using RDatasets
using Statistics

include("iris_binary.jl")
include("iris.jl")
include("alligators.jl")

@info "$(now()) Finished"