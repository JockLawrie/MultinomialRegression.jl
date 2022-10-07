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
    B      = coef(model)
    pmean_true = 0.9631165919665698
    Btrue = [-42.637803644985866; -2.465220193358506;  -6.680886997622709; 9.429385124377749; 18.286136839432427;;]
    @test isapprox(pmean, pmean_true; atol=1e-10)
    @test isapprox(B, Btrue; atol=1e-10)
    @test !isregularized(model)

end