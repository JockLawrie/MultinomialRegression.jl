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
ptrue  = 0.975388265191401
Btrue  = [7.655763785408953 -34.980132384842996; -1.4800592740561043 -3.9457712168326466; -7.521061777314869 -14.203413238492303; 5.602142924108288 15.032509306476685; 7.824993058749435 26.113944534250077]
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
ptrue  = 0.9337207895812999
Btrue  = [9.935596018734074e-7 -4.194685532114717; -0.10179174405956715 -2.71479536273949; -3.263336715859761 -6.162331485302718; 4.693254729167252 8.597898821307428; 2.4460784087100167e-7 5.842147921033004]
@test isapprox(pmean, ptrue; atol=1e-8)
@test isapprox(coef(model), Btrue; atol=1e-8)
@test isregularized(model)

# L2 regularized fit
model  = fit(@formula(Species ~ 1 + SepalLength + SepalWidth + PetalLength + PetalWidth), iris; reg=L2(0.5))
iris.p = [predict(model, X[i, :]) for i = 1:nrow(iris)]
pmean  = mean([iris.p[i][y[i]] for i = 1:nrow(iris)])
ptrue  = 0.8709932478191903
Btrue  = [0.7888393556722058 -1.8380728122477037; -0.19997312717019006 -1.9817654638534465; -1.8211924740229675 -3.0788414908935984; 2.184073670058001 4.507890041566887; -0.23844730696797464 3.4034321175700617]
@test isapprox(pmean, ptrue; atol=1e-8)
@test isapprox(coef(model), Btrue; atol=1e-8)
@test isregularized(model)

# ElasticNet regularized fit
model  = fit(@formula(Species ~ 1 + SepalLength + SepalWidth + PetalLength + PetalWidth), iris; reg=ElasticNet(0.5, 0.5))
iris.p = [predict(model, X[i, :]) for i = 1:nrow(iris)]
pmean  = mean([iris.p[i][y[i]] for i = 1:nrow(iris)])
ptrue  = 0.8550458454192683
Btrue  = [0.00028742832791726626 -1.3677000185370674; -9.320559348425432e-6 -1.8152969063402398; -1.7527629918053778 -2.88057882227795; 2.0099374514627626 4.198568250100472; 0.000514315230815811 3.1938398322674098]
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
Btrue = [8.896968057652046 -32.397705940531644; -1.795711057394225 -4.476180261259748; -7.073489617803958 -13.026831561488326; 5.510095704333738 15.195225300686422; 7.390454234913102 23.947696520620806]
@test isapprox(coef(model), Btrue; atol=1e-8)

# Test that weights are scaled to sum to the number of observations
w2    = 2*w
model = fit(@formula(Species ~ 1 + SepalLength + SepalWidth + PetalLength + PetalWidth), iris; wts=w2)
@test isapprox(coef(model), Btrue; atol=1e-8)

end