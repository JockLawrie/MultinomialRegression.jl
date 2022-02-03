@testset "Iris" begin

@info "$(now()) Starting test set: Iris"

# Data
iris = dataset("datasets", "iris")
iris.intercept = fill(1.0, nrow(iris))
y = iris.Species.refs
X = Matrix(iris[:, ["intercept", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]])

# Unregularized fit
fitted = fit(y, X)
iris.p = [predict(fitted, X[i, :]) for i = 1:nrow(iris)]
pmean  = mean([iris.p[i][y[i]] for i = 1:nrow(iris)])
B2     = [15.619498241390401 -27.018305571964458; -9.379861599935365 -11.84508179488949; -23.8502281938097 -30.531115207983206; 36.95799897683483 46.38738413069437; 10.435998075531359 28.722134963206237]
@test isapprox(pmean, 0.9754110613097936; atol=1e-10)
@test isapprox(coef(fitted), B2; atol=1e-10)
@test !isregularized(fitted)

# L1 regularized fit
fitted = fit(y, X, L1(0.5))
iris.p = [predict(fitted, X[i, :]) for i = 1:nrow(iris)]
pmean  = mean([iris.p[i][y[i]] for i = 1:nrow(iris)])
B2     = [8.025985174126068e-16 -4.13595102341692; -6.188251901210096e-16 -2.5938791586482894; -2.503084820591207 -5.428856394912443; 2.7933554949572197 6.6790919484457385; 2.2249098456178565e-15 5.8394611372133545]
@test isapprox(pmean, 0.9296851911723949; atol=1e-10)
@test isapprox(coef(fitted), B2; atol=1e-10)
@test isregularized(fitted)

# L2 regularized fit
fitted = fit(y, X, L2(0.5))
iris.p = [predict(fitted, X[i, :]) for i = 1:nrow(iris)]
pmean  = mean([iris.p[i][y[i]] for i = 1:nrow(iris)])
B2     = [0.7888393490091408 -1.8380728113497204; -0.19997312515913346 -1.9817654634065198; -1.8211924718875678 -3.078841490380221; 2.184073665201137 4.507890037838481; -0.23844731040849787 3.403432115649799]
@test isapprox(pmean, 0.8709932476955314; atol=1e-10)
@test isapprox(coef(fitted), B2; atol=1e-10)
@test isregularized(fitted)

end