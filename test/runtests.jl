using Test
using MultinomialRegression
const MultinomialRegression = MultinomialRegression  # Clashes with MLJLinearModels
const fit = MultinomialRegression.fit  # Clashes with MLJLinearModels

using DataFrames
using Dates
using Logging
using MLJLinearModels
using RDatasets
using Random: randperm
using StableRNGs
using Statistics

function mlj_fit(y, X; opts...)
    reg = MLJLinearModels.MultinomialRegression(; opts...)
    b   = MLJLinearModels.fit(reg, X, y)
    nx  = size(X, 2)
    c   = floor(Int, length(b) / nx)
    B   = reshape(b, nx, c)  # (p + 1) x c
end

function mlj_predict(X, B)
    n = size(X, 1)
    c = size(B, 2)
    result = exp.(X * B)  # n x c
    for i = 1:n
        s = sum(result[i, :])
        for j = 1:c
            result[i, j] /= s
        end
    end
    result
end

include("iris.jl")
include("compare_to_MLJLinearModels.jl")

@info "$(now()) Finished"