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

# L1 regularized fit
fitted = fit(y, X, L1(0.5))
iris.p = [predict(fitted, X[i, :]) for i = 1:nrow(iris)]
pmean  = mean([iris.p[i][y[i]] for i = 1:nrow(iris)])
B2     = [1.1337032026413356e-13 -4.1595656797224; -3.3517374716573104e-14 -2.592612567654327; -2.5053234563377544 -5.433956245273727; 2.7956161747325914 6.684344249687155; 1.9299797966004742e-14 5.844993378673247]
@test isapprox(pmean, 0.9298076683272863; atol=1e-10)
@test isapprox(coef(fitted), B2; atol=1e-10)

# L2 regularized fit
fitted = fit(y, X, L2(0.5))
iris.p = [predict(fitted, X[i, :]) for i = 1:nrow(iris)]
pmean  = mean([iris.p[i][y[i]] for i = 1:nrow(iris)])
B2     = [0.7888393490091408 -1.8380728113497204; -0.19997312515913346 -1.9817654634065198; -1.8211924718875678 -3.078841490380221; 2.184073665201137 4.507890037838481; -0.23844731040849787 3.403432115649799]
@test isapprox(pmean, 0.8709932476955314; atol=1e-10)
@test isapprox(coef(fitted), B2; atol=1e-10)

# Box regularized fit (constrain each parameter to be with [lowerbound, upperbound])
fitted = fit(y, X, BoxRegularizer(-5.0, 5.0))
iris.p = [predict(fitted, X[i, :]) for i = 1:nrow(iris)]
pmean  = mean([iris.p[i][y[i]] for i = 1:nrow(iris)])
B2     = [-1.154269694381064 -2.763370935316265; 0.49175393083592134 -1.6891482151450825; -2.465312829491497 -3.5685227107246056; 2.390464481023911 4.6568814147255075; -1.338245409454034 3.098099899311478]
@test isapprox(pmean, 0.8766776510049568; atol=1e-10)
@test isapprox(coef(fitted), B2; atol=1e-10)

end