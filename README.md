# MultinomiaRegression.jl

```julia
using MultinomialRegression

using DataFrames
using RDatasets
using Statistics

# Data
iris = dataset("datasets", "iris")
iris.intercept = fill(1.0, nrow(iris))
y = iris.Species.refs
X = Matrix(iris[:, [1,2,3,4,6]])

# Unregularised fit
B = fit(y, X)
iris.p = [predict(B, X[i, :]) for i = 1:nrow(iris)]  # Each prediction is a vector of probabilities
mean([iris.p[i][y[i]] for i = 1:nrow(iris)])         # Mean Pr(Y = y[i])

B2 = [-7.927818200468233 -10.393072311241966; -18.70033161617695 -25.38119065922222; 26.76422074394354 36.19365886259815; 5.558371894028679 23.844497063952385; 16.716166099230357 -25.92176161756603]
pmean = mean([iris.p[i][y[i]] for i = 1:nrow(iris)])

isapprox(pmean, 0.9754110790815228; atol=1e-10)  # Reproducible result
isapprox(B, B2; atol=1e-10)  # Reproducible result

# L2 regularised fit
B      = fit(y, X, L2(0.5))
iris.p = [predict(B, X[i, :]) for i = 1:nrow(iris)]
pmean  = mean([iris.p[i][y[i]] for i = 1:nrow(iris)])

B2 = [-0.2003117733232103 -1.9810446456446502; -1.8218221042649223 -3.080450064972763; 2.1852233068431217 4.507699288823395; -0.24007026523207944 3.4037938446516254; 0.7910229216904574 -1.835692080815499]
isapprox(pmean, 0.8710102985804118; atol=1e-10)
isapprox(B, B2; atol=1e-10)
```