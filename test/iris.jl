@testset "Iris" begin

@info "$(now()) Starting test set: Iris"

# Data
iris = dataset("datasets", "iris")
iris.intercept = fill(1.0, nrow(iris))
y = iris.Species.refs
X = Matrix(iris[:, ["intercept", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]])

# Unregularized fit
model  = fit(@formula(Species ~ 1 + SepalLength + SepalWidth + PetalLength + PetalWidth), iris)
iris.p = [predict(model, X[i, :]) for i = 1:nrow(iris)]
pmean  = mean([iris.p[i][y[i]] for i = 1:nrow(iris)])
B      = coef(model)
ptrue  = 0.9749774974660096
Btrue  = [4.975415294679369 -37.66545139525668; -0.5700492887839999 -3.03391914896401; -5.792929082291398 -12.471480403747798; 4.159722869456368 13.5869695098448; 3.8991764651585226 22.179722316756568]
@test isapprox(pmean, ptrue; atol=1e-10)
@test isapprox(B, Btrue; atol=1e-10)
@test !isregularized(model)

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

# Access values
B["(Intercept)", "virginica"]
B["(Intercept)", :]
B[:, "virginica"]
B[1, 2]

# L1 regularized fit
model  = fit(@formula(Species ~ 1 + SepalLength + SepalWidth + PetalLength + PetalWidth), iris; reg=L1(0.5))
iris.p = [predict(model, X[i, :]) for i = 1:nrow(iris)]
pmean  = mean([iris.p[i][y[i]] for i = 1:nrow(iris)])
ptrue  = 0.9298225328925666
Btrue  = [2.8865305259902325e-18 -4.164518643934148; -4.0003098839556167e-16 -2.6063230765004057; -2.5073901642945966 -5.418536097974986; 2.7968458170514072 6.700398130596685; -3.1946541988440898e-18 5.82486558126391]
@test isapprox(pmean, ptrue; atol=1e-10)
@test isapprox(coef(model), Btrue; atol=1e-10)
@test isregularized(model)

# L2 regularized fit
model  = fit(@formula(Species ~ 1 + SepalLength + SepalWidth + PetalLength + PetalWidth), iris; reg=L2(0.5))
iris.p = [predict(model, X[i, :]) for i = 1:nrow(iris)]
pmean  = mean([iris.p[i][y[i]] for i = 1:nrow(iris)])
ptrue  = 0.8709932476955317
Btrue  = [0.7888393490091279 -1.838072811349677; -0.19997312515915025 -1.9817654634065525; -1.821192471887549 -3.0788414903801886; 2.1840736652011756 4.507890037838502; -0.2384473104085673 3.4034321156497853]
@test isapprox(pmean, ptrue; atol=1e-10)
@test isapprox(coef(model), Btrue; atol=1e-10)
@test isregularized(model)

# ElasticNet regularized fit
model  = fit(@formula(Species ~ 1 + SepalLength + SepalWidth + PetalLength + PetalWidth), iris; reg=ElasticNet(0.5, 0.5))
iris.p = [predict(model, X[i, :]) for i = 1:nrow(iris)]
pmean  = mean([iris.p[i][y[i]] for i = 1:nrow(iris)])
ptrue  = 0.851627021342546
Btrue  = [2.202285662861181e-20 -1.352689711252036; -9.48635015171177e-7 -1.826793038733544; -1.6811170026719975 -2.8229214666028595; 1.8645059515489941 4.08103262615276; -1.6624667692556117e-7 3.1734225402322225]
@test isapprox(pmean, ptrue; atol=1e-10)
@test isapprox(coef(model), Btrue; atol=1e-10)
@test isregularized(model)

# Weighted fit with each weight equal to 1
w     = fill(1.0, 150)
model = fit(@formula(Species ~ 1 + SepalLength + SepalWidth + PetalLength + PetalWidth), iris; wts=w)
@test isapprox(coef(model), B; atol=1e-10)

# Weighted fit with a mix of weights
w = collect(0.25:0.01:1.75)
splice!(w, findfirst(==(1.0), w))
model = fit(@formula(Species ~ 1 + SepalLength + SepalWidth + PetalLength + PetalWidth), iris; wts=w)
Btrue = [6.035229502487764 -35.264648153156564; -0.848901014197533 -3.527483061010013; -5.3482096886032755 -11.297721710725307; 4.122088167195583 13.804194657706852; 3.297230688297391 19.845794726325643]
@test isapprox(coef(model), Btrue; atol=1e-10)

# Test that weights are scaled to sum to the number of observations
w2    = 2*w
model = fit(@formula(Species ~ 1 + SepalLength + SepalWidth + PetalLength + PetalWidth), iris; wts=w2)
@test isapprox(coef(model), Btrue; atol=1e-10)

end