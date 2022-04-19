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
B2     = [15.619498241390401 -27.018305571964458; -9.379861599935365 -11.84508179488949; -23.8502281938097 -30.531115207983206; 36.95799897683483 46.38738413069437; 10.435998075531359 28.722134963206237]
@test isapprox(pmean, 0.9754110613097936; atol=1e-10)
@test isapprox(B, B2; atol=1e-10)
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
B2     = [8.025985174126068e-16 -4.13595102341692; -6.188251901210096e-16 -2.5938791586482894; -2.503084820591207 -5.428856394912443; 2.7933554949572197 6.6790919484457385; 2.2249098456178565e-15 5.8394611372133545]
@test isapprox(pmean, 0.9296851911723949; atol=1e-10)
@test isapprox(coef(model), B2; atol=1e-10)
@test isregularized(model)

# L2 regularized fit
model  = fit(@formula(Species ~ 1 + SepalLength + SepalWidth + PetalLength + PetalWidth), iris; reg=L2(0.5))
iris.p = [predict(model, X[i, :]) for i = 1:nrow(iris)]
pmean  = mean([iris.p[i][y[i]] for i = 1:nrow(iris)])
B2     = [0.7888393490091408 -1.8380728113497204; -0.19997312515913346 -1.9817654634065198; -1.8211924718875678 -3.078841490380221; 2.184073665201137 4.507890037838481; -0.23844731040849787 3.403432115649799]
@test isapprox(pmean, 0.8709932476955314; atol=1e-10)
@test isapprox(coef(model), B2; atol=1e-10)
@test isregularized(model)

# ElasticNet regularized fit
model  = fit(@formula(Species ~ 1 + SepalLength + SepalWidth + PetalLength + PetalWidth), iris; reg=ElasticNet(0.5, 0.5))
iris.p = [predict(model, X[i, :]) for i = 1:nrow(iris)]
pmean  = mean([iris.p[i][y[i]] for i = 1:nrow(iris)])
B2     = [1.852884572118782e-21 -1.3525808490106175; -1.4235665330836348e-10 -1.8282508130541715; -1.6813170037288188 -2.8222993906240093; 1.8646923442188061 4.082829660706014; -7.00894936397621e-10 3.1726435363590673]
@test isapprox(pmean, 0.8516537261832; atol=1e-10)
@test isapprox(coef(model), B2; atol=1e-10)
@test isregularized(model)

# Weighted fit
w = collect(0.25:0.01:1.75)
splice!(w, findfirst(==(1.0), w))
model = fit(@formula(Species ~ 1 + SepalLength + SepalWidth + PetalLength + PetalWidth), iris; wts=w)
B2    = [16.127559313455407 -25.16921609276914; -7.689854268602372 -10.369812798632715; -17.193729922640664 -23.145562841624315; 26.836554931444297 36.520710594561244; 6.851897121946295 23.40631253745369]
@test isapprox(coef(model), B2; atol=1e-10)

# Test that weights are scaled to sum to the number of observations
w2    = 2*w
model = fit(@formula(Species ~ 1 + SepalLength + SepalWidth + PetalLength + PetalWidth), iris; wts=w2)
@test isapprox(coef(model), B2; atol=1e-10)

end