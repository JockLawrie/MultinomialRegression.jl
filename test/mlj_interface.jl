@testset "MLJ Interface" begin

using MLJModelInterface

iris = dataset("datasets", "iris")
iris.intercept = fill(1.0, nrow(iris))
xnames = ["intercept", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]
X = Matrix(iris[:, xnames])
y = iris.Species
mach = MultinomialLogisticRegressor(; yname="Species", xnames=xnames)
model, cache, report = fit(mach, 0, X, y)

B     = coef(model)
Btrue = [4.975415294660627 -37.66545139528325; -0.5700492887738026 -3.0339191489509885; -5.792929082301356 -12.471480403756624; 4.159722869451328 13.586969509837358; 3.899176465164596 22.179722316762103]
@test isapprox(B, Btrue; atol=1e-10)

Xnew = X[1:5, :]
println(predict(mach, model, Xnew))
Xnew = X[1, :]
println(predict(mach, model, Xnew))

end
