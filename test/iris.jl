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
ptrue  = 0.929853979053618
Btrue  = [2.335287866989595e-15 -4.198253238662058; -9.519040699058266e-16 -2.5870714529107075; -2.506972190209195 -5.428750790324113; 2.7964843658568928 6.683577923842485; 4.5622739762807e-16 5.839931381703964]
@test isapprox(pmean, ptrue; atol=1e-10)
@test isapprox(coef(model), Btrue; atol=1e-10)
@test isregularized(model)

# L2 regularized fit
model  = fit(@formula(Species ~ 1 + SepalLength + SepalWidth + PetalLength + PetalWidth), iris; reg=L2(0.5))
iris.p = [predict(model, X[i, :]) for i = 1:nrow(iris)]
pmean  = mean([iris.p[i][y[i]] for i = 1:nrow(iris)])
ptrue  = 0.8709932476955317
Btrue  = [0.7888393490091464 -1.838072811349745; -0.19997312515924456 -1.981765463406657; -1.8211924718874102 -3.0788414903800367; 2.1840736652011583 4.507890037838522; -0.23844731040845246 3.4034321156498524]
@test isapprox(pmean, ptrue; atol=1e-10)
@test isapprox(coef(model), Btrue; atol=1e-10)
@test isregularized(model)

# ElasticNet regularized fit
model  = fit(@formula(Species ~ 1 + SepalLength + SepalWidth + PetalLength + PetalWidth), iris; reg=ElasticNet(0.5, 0.5))
iris.p = [predict(model, X[i, :]) for i = 1:nrow(iris)]
pmean  = mean([iris.p[i][y[i]] for i = 1:nrow(iris)])
ptrue  = 0.8516356639568854
Btrue  = [1.164670302474663e-21 -1.3531544411532592; 3.2278532459425955e-9 -1.827088055116832; -1.6813436609110648 -2.8228310610351546; 1.8647858975392821 4.082005309230107; 4.998371884461643e-11 3.172220120680883]
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
Btrue = [6.035229502503111 -35.26464815315409; -0.8489010142050419 -3.5274830610109134; -5.348209688597931 -11.2977217107236; 4.122088167203922 13.80419465770622; 3.29723068828238 19.84579472632648]
@test isapprox(coef(model), Btrue; atol=1e-10)

# Test that weights are scaled to sum to the number of observations
w2    = 2*w
model = fit(@formula(Species ~ 1 + SepalLength + SepalWidth + PetalLength + PetalWidth), iris; wts=w2)
@test isapprox(coef(model), Btrue; atol=1e-10)

end