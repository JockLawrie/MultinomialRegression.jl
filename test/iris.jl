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
ptrue  = 0.9753882651914038
Btrue  = [7.655763785129046 -34.98013238513413; -1.480059273832724 -3.9457712166099057; -7.521061777586373 -14.203413238765094; 5.602142923974062 15.03250930634506; 7.824993058959534 26.113944534463773]
@test isapprox(pmean, ptrue; atol=1e-8)
@test isapprox(B, Btrue; atol=1e-8)
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
ptrue  = 0.9337207908300835
Btrue  = [1.0812859676802515e-6 -4.19468624044427; -0.1017918498645732 -2.7147952942838582; -3.2633366156264816 -6.162331340696932; 4.693254879696026 8.597898785623677; -1.1887985352655251e-8 5.842147960298112]
@test isapprox(pmean, ptrue; atol=1e-8)
@test isapprox(coef(model), Btrue; atol=1e-8)
@test isregularized(model)

# L2 regularized fit
model  = fit(@formula(Species ~ 1 + SepalLength + SepalWidth + PetalLength + PetalWidth), iris; reg=L2(0.5))
iris.p = [predict(model, X[i, :]) for i = 1:nrow(iris)]
pmean  = mean([iris.p[i][y[i]] for i = 1:nrow(iris)])
ptrue  = 0.8709932476132374
Btrue  = [0.7888393496647931 -1.8380728122058165; -0.19997312540505394 -1.9817654639004803; -1.8211924695771182 -3.0788414880170656; 2.184073660322796 4.507890034456721; -0.23844730575302153 3.4034321176938698]
@test isapprox(pmean, ptrue; atol=1e-8)
@test isapprox(coef(model), Btrue; atol=1e-8)
@test isregularized(model)

# ElasticNet regularized fit
model  = fit(@formula(Species ~ 1 + SepalLength + SepalWidth + PetalLength + PetalWidth), iris; reg=ElasticNet(0.5, 0.5))
iris.p = [predict(model, X[i, :]) for i = 1:nrow(iris)]
pmean  = mean([iris.p[i][y[i]] for i = 1:nrow(iris)])
ptrue  = 0.8550458454192695
Btrue  = [0.0002874283279182458 -1.3677000185370765; -9.320559346084626e-6 -1.815296906340264; -1.752762991805398 -2.8805788222779016; 2.009937451462787 4.198568250100478; 0.0005143152308182155 3.1938398322674564]
@test isapprox(pmean, ptrue; atol=1e-8)
@test isapprox(coef(model), Btrue; atol=1e-8)
@test isregularized(model)

# Weighted fit with each weight equal to 1
w     = fill(1.0, 150)
model = fit(@formula(Species ~ 1 + SepalLength + SepalWidth + PetalLength + PetalWidth), iris; wts=w)
@test isapprox(coef(model), B; atol=1e-8)

# Weighted fit with a mix of weights
w = collect(0.25:0.01:1.75)
splice!(w, findfirst(==(1.0), w))
model = fit(@formula(Species ~ 1 + SepalLength + SepalWidth + PetalLength + PetalWidth), iris; wts=w)
Btrue = [8.896968058935158 -32.39770593923794; -1.7957110576820703 -4.476180261548935; -7.07348961791294 -13.026831561598048; 5.510095704600754 15.195225300954261; 7.390454234611988 23.947696520317113]
@test isapprox(coef(model), Btrue; atol=1e-8)

# Test that weights are scaled to sum to the number of observations
w2    = 2*w
model = fit(@formula(Species ~ 1 + SepalLength + SepalWidth + PetalLength + PetalWidth), iris; wts=w2)
@test isapprox(coef(model), Btrue; atol=1e-8)

end