# MultinomialRegression.jl

```julia
using DataFrames
using MultinomialRegression
using RDatasets
using Statistics

# Data
iris = dataset("datasets", "iris")
iris.intercept = fill(1.0, nrow(iris))
yname   = "Species"
ylevels = levels(iris.Species)
xnames  = ["intercept", "SepalWidth"]
y = iris.Species.refs
X = Matrix(iris[:, xnames])

# Unregularized fit
model = fit(y, X, yname, ylevels, xnames)  # fit(y, X) works too

# Model-level diagnostics
isregularized(model)
nobs(model)
loglikelihood(model)
aic(model)
aicc(model)
bic(model)

# Coefficient-level diagnostics
nparams(model)
coef(model)
stderror(model)
coeftable(model)
vcov(model)
coefcor(model)

# Get values using row names and column names
B = coef(model)
B["intercept", "virginica"]
B["intercept", :]
B[:, "virginica"]
B[1, 2]  # Integer indices also work

B2 = [18.858436609230726 12.997324400573637; -6.118961539544502 -4.079098098175112]
isapprox(B, B2; atol=1e-10)  # Reproducible result

#=
  Regularized fit
  Note: Standard errors not yet available because the current method requires that the 
        parameters are maximum likelihood estimates, which regularized parameters are usually not.
=#
model_L1 = fit(y, X, yname, ylevels, xnames, L1(0.5))
model_L2 = fit(y, X, yname, ylevels, xnames, L2(0.5))
```