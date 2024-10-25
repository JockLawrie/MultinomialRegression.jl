@testset "Iris - binary" begin

@info "$(now()) Starting test set: Iris - binary"

# Data
iris = dataset("datasets", "iris")
iris = iris[iris.Species .!= "setosa", :]
iris[!, "species_binary"] = [x == "versicolor" ? 1 : 2 for x in iris.Species]
iris.intercept = fill(1.0, nrow(iris))

# Unregularized fit
model  = fit(@formula(species_binary ~ 1 + SepalLength + SepalWidth + PetalLength + PetalWidth), iris)
y      = iris.species_binary
X      = Matrix(iris[:, ["intercept", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]])
iris.p = [predict(model, X[i, :]) for i = 1:nrow(iris)]
pmean  = mean([iris.p[i][y[i]] for i = 1:nrow(iris)])
pmean_true = 0.9631165916907815
Btrue = [-42.637802709142136; -2.465220217741298; -6.680886821118824; 9.429384998498609; 18.286136443996313;;]
@test isapprox(pmean, pmean_true; atol=1e-8)
@test isapprox(coef(model), Btrue; atol=1e-8)
@test !isregularized(model)

end