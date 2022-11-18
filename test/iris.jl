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
Btrue  = [4.975415294660627 -37.66545139528325; -0.5700492887738026 -3.0339191489509885; -5.792929082301356 -12.471480403756624; 4.159722869451328 13.586969509837358; 3.899176465164596 22.179722316762103]
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
ptrue  = 0.9297154864981869
Btrue  = [6.072406712993603e-16 -4.170313920372684; -5.103571874171169e-16 -2.5926771270758193; -2.5047512205520848 -5.42004928714557; 2.7957520824892623 6.687224293714904; 4.495308997872163e-16 5.820169395590656]
@test isapprox(pmean, ptrue; atol=1e-10)
@test isapprox(coef(model), Btrue; atol=1e-10)
@test isregularized(model)

# L2 regularized fit
model  = fit(@formula(Species ~ 1 + SepalLength + SepalWidth + PetalLength + PetalWidth), iris; reg=L2(0.5))
iris.p = [predict(model, X[i, :]) for i = 1:nrow(iris)]
pmean  = mean([iris.p[i][y[i]] for i = 1:nrow(iris)])
ptrue  = 0.8709932476955317
Btrue  = [0.7888393490091479 -1.8380728113497076; -0.19997312515912738 -1.9817654634065132; -1.8211924718875832 -3.0788414903802384; 2.184073665201141 4.507890037838479; -0.23844731040850708 3.4034321156498044]
@test isapprox(pmean, ptrue; atol=1e-10)
@test isapprox(coef(model), Btrue; atol=1e-10)
@test isregularized(model)

# ElasticNet regularized fit
model  = fit(@formula(Species ~ 1 + SepalLength + SepalWidth + PetalLength + PetalWidth), iris; reg=ElasticNet(0.5, 0.5))
iris.p = [predict(model, X[i, :]) for i = 1:nrow(iris)]
pmean  = mean([iris.p[i][y[i]] for i = 1:nrow(iris)])
ptrue  = 0.8515497075497324
Btrue  = [1.0842021724855044e-19 -1.3231506716022616; 3.0586173883587284e-6 -1.8217681861275952; -1.6815106753637108 -2.838254969678851; 1.864380216740523 4.0752418192938915; -7.153043454269825e-6 3.17940953616683]
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
Btrue = [6.0352295025676845 -35.26464815308396; -0.8489010142193001 -3.5274830610284504; -5.348209688603029 -11.297721710726185; 4.122088167214936 13.804194657721203; 3.297230688272863 19.84579472630977]
@test isapprox(coef(model), Btrue; atol=1e-10)

# Test that weights are scaled to sum to the number of observations
w2    = 2*w
model = fit(@formula(Species ~ 1 + SepalLength + SepalWidth + PetalLength + PetalWidth), iris; wts=w2)
@test isapprox(coef(model), Btrue; atol=1e-10)

end