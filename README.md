# MultinomialRegression.jl

```julia
using MultinomialRegression
using DataFrames
using RDatasets

# Data
iris = dataset("datasets", "iris")

# Unregularized fit
model = fit(@formula(Species ~ 1 + SepalWidth), iris)

# Predict
xnew = [1.0, iris.SepalWidth[1]]
pred = predict(model, xnew)

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
B["(Intercept)", "virginica"]
B["(Intercept)", :]
B[:, "virginica"]
B[1, 2]  # Integer indices also work

B2 = [18.858436609230726 12.997324400573637; -6.118961539544502 -4.079098098175112]
isapprox(B, B2; atol=1e-10)  # Reproducible result

#=
  Regularized fit
  Note: Standard errors not yet available because the current method requires that the 
        parameters are maximum likelihood estimates, which regularized parameters usually aren't.
=#
model_L1 = fit(@formula(Species ~ 1 + SepalWidth), iris, L1(0.5))
model_L2 = fit(@formula(Species ~ 1 + SepalWidth), iris, L2(0.5))
```