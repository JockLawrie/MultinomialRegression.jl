# MultinomialRegression.jl

```julia
using MultinomialRegression
using DataFrames
using RDatasets

# Data
iris = dataset("datasets", "iris")

# Unregularized fit
model = fit(@formula(Species ~ 1 + SepalWidth), iris)
opts  = Dict(:iterations => 250, :f_abstol => 1e-9)  # Same terminology as Optim.Options
model = fit(@formula(Species ~ 1 + SepalWidth), iris; opts=opts)

# Predict
xnew  = [1.0, iris.SepalWidth[1]]
pred  = predict(model, xnew)
pred2 = zeros(3)
predict!(pred2, model, xnew)

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
coeftable(model)
stderror(model)
vcov(model)
coefcor(model)

# Get values using row names and column names
B = coef(model)
B["(Intercept)", "virginica"]
B["(Intercept)", :]
B[:, "virginica"]
B[1, 2]  # Integer indices also work

B2 = [18.858251759055936 12.997166560384427; -6.118905216314177 -4.079050714520645]
isapprox(B, B2; atol=1e-10)  # Reproducible result

#=
  Regularized fit
  Note: Standard errors calculated from the hessian evaluated at the parameters.
        Since the parameters of regularized models are not MLEs, the standard errors are approximate.
=#
model_L1    = fit(@formula(Species ~ 1 + SepalWidth), iris; reg=L1(0.5))
model_L2    = fit(@formula(Species ~ 1 + SepalWidth), iris; reg=L2(0.5))
model_ElNet = fit(@formula(Species ~ 1 + SepalWidth), iris; reg=ElasticNet(0.5, 0.5))

# Weighted fit
w = collect(0.25:0.01:1.75)
splice!(w, findfirst(==(1.0), w))
weighted_fit = fit(@formula(Species ~ 1 + SepalWidth), iris; wts=w)

# Under the hood, weights are scaled to sum to the number of observations
w2 = 2*w
weighted_fit2 = fit(@formula(Species ~ 1 + SepalWidth), iris; wts=w2)
isapprox(coef(weighted_fit2), coef(weighted_fit); atol=1e-10)
```