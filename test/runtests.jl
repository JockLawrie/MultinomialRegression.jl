using Test
using MultinomialRegression

using DataFrames
using RDatasets
using Statistics

# Data
iris = dataset("datasets", "iris")
y = iris.Species.refs
X = Matrix(iris[:, 1:4])

# Fit, predict
B = fit(y, X)
iris.p = [predict(B, X[i, :]) for i = 1:nrow(iris)]

# Test
B2 = [-1.7807813740963174 -8.108816596211382; -8.639705665839527 -15.258289407312507; 12.877899753268665 21.31207276071631; 2.8383123741777703 13.121630584586427]
pmean = mean([iris.p[i][y[i]] for i = 1:nrow(iris)])

@test isapprox(pmean, 0.9572778596997049; atol=1e-8)
@test isapprox(B, B2; atol=1e-8)