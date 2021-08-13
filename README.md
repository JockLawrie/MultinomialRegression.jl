# MultinomialRegression.jl

```julia
using DataFrames
using MultinomialRegression
using RDatasets
using Statistics

# Data
iris = dataset("datasets", "iris")
iris.intercept = fill(1.0, nrow(iris))
y = iris.Species.refs
X = Matrix(iris[:, ["intercept", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]])

# Unregularized fit
B      = fit(y, X)
iris.p = [predict(B, X[i, :]) for i = 1:nrow(iris)]    # Each prediction is a vector of probabilities
pmean  = mean([iris.p[i][y[i]] for i = 1:nrow(iris)])  # Mean Pr(Y = y[i])
B2     = [12.492550959964307 -30.145511838513336; -13.805841696210813 -16.27098488581211; -39.46413451928773 -46.145154218613975; 61.59580691999151 71.02512875924981; 21.499505874304305 39.78592272496369]
@test isapprox(pmean, 0.9754110798138272; atol=1e-10)  # Reproducible result
@test isapprox(B, B2; atol=1e-10)  # Reproducible result

# L1 regularized fit
B      = fit(y, X, L1(0.5))
iris.p = [predict(B, X[i, :]) for i = 1:nrow(iris)]
pmean  = mean([iris.p[i][y[i]] for i = 1:nrow(iris)])
B2     = [-2.176030356478348e-5 -3.368372096445269; -4.6375342534571835e-5 -2.659295361335591; -2.5407374384134696 -5.712585768725875; 2.829164602857422 6.800714978069462; 3.628185714330291e-5 5.798698371948723]
@test isapprox(pmean, 0.9293218278737707; atol=1e-10)
@test isapprox(B, B2; atol=1e-10)

# L2 regularized fit
B      = fit(y, X, L2(0.5))
iris.p = [predict(B, X[i, :]) for i = 1:nrow(iris)]
pmean  = mean([iris.p[i][y[i]] for i = 1:nrow(iris)])
B2     = [0.7945515804552191 -1.8331489499819589; -0.19925209776764655 -1.9809068067370361; -1.8239979618578914 -3.0818164583764194; 2.18416811002322 4.507783033080395; -0.24055279601414303 3.4020416953749906]
@test isapprox(pmean, 0.8710024596383066; atol=1e-10)
@test isapprox(B, B2; atol=1e-10)

# Box regularized fit (constrain each parameter to be with [lowerbound, upperbound])
B      = fit(y, X, BoxRegularizer(-10.0, 10.0))
iris.p = [predict(B, X[i, :]) for i = 1:nrow(iris)]
pmean  = mean([iris.p[i][y[i]] for i = 1:nrow(iris)])
B2     = [9.998861432630893 -9.999860147735738; -0.8235495447737922 -2.863978840177503; -6.173999928766744 -9.99999986003335; 4.591535394661463 9.999999999998991; -0.2467962531036445 9.999999999492559]
@test isapprox(pmean, 0.9642848030231086; atol=1e-10)
@test isapprox(B, B2; atol=1e-10)
```