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
ptrue  = 0.929774513412857
Btrue  = [6.61753695119197e-16 -4.176864480798235; 1.5933749647626836e-15 -2.5939044704413847; -2.5055342869998336 -5.423021292307201; 2.7962929512649044 6.690862400042932; -5.138087602230686e-16 5.8233708609115915]
@test isapprox(pmean, ptrue; atol=1e-10)
@test isapprox(coef(model), Btrue; atol=1e-10)
@test isregularized(model)

# L2 regularized fit
model  = fit(@formula(Species ~ 1 + SepalLength + SepalWidth + PetalLength + PetalWidth), iris; reg=L2(0.5))
iris.p = [predict(model, X[i, :]) for i = 1:nrow(iris)]
pmean  = mean([iris.p[i][y[i]] for i = 1:nrow(iris)])
ptrue  = 0.8709932476955317
Btrue  = [0.7888393490091519 -1.8380728113497027; -0.19997312515911816 -1.9817654634065034; -1.8211924718875934 -3.078841490380243; 2.184073665201134 4.507890037838466; -0.23844731040851266 3.403432115649804]
@test isapprox(pmean, ptrue; atol=1e-10)
@test isapprox(coef(model), Btrue; atol=1e-10)
@test isregularized(model)

# ElasticNet regularized fit
model  = fit(@formula(Species ~ 1 + SepalLength + SepalWidth + PetalLength + PetalWidth), iris; reg=ElasticNet(0.5, 0.5))
iris.p = [predict(model, X[i, :]) for i = 1:nrow(iris)]
pmean  = mean([iris.p[i][y[i]] for i = 1:nrow(iris)])
ptrue  = 0.8516973166463138
Btrue  = [7.623296525288703e-21 -1.3433769673934814; -1.3347616049958702e-7 -1.8304908217392464; -1.6818067853646446 -2.825605571512041; 1.8655917519561591 4.08715046967263; 4.61893712963144e-7 3.1703215924297736]
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
Btrue = [6.035229502567349 -35.264648153084146; -0.8489010142192825 -3.527483061028543; -5.348209688602884 -11.297721710726172; 4.122088167214877 13.804194657721421; 3.297230688272912 19.845794726309606]
@test isapprox(coef(model), Btrue; atol=1e-10)

# Test that weights are scaled to sum to the number of observations
w2    = 2*w
model = fit(@formula(Species ~ 1 + SepalLength + SepalWidth + PetalLength + PetalWidth), iris; wts=w2)
@test isapprox(coef(model), Btrue; atol=1e-10)

end