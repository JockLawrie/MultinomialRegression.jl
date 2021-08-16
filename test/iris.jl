@testset "Iris" begin

@info "$(now()) Starting test set: Iris"

# Data
iris = dataset("datasets", "iris")
iris.intercept = fill(1.0, nrow(iris))
y = iris.Species.refs
X = Matrix(iris[:, ["intercept", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]])

# Unregularized fit
B0     = fit(y, X)
iris.p = [predict(B0, X[i, :]) for i = 1:nrow(iris)]
pmean  = mean([iris.p[i][y[i]] for i = 1:nrow(iris)])
B2     = [15.733603483202188 -26.904200315041635; -9.200380133712661 -11.665600329353945; -23.32356217434708 -30.00444918863351; 36.02807433495139 45.45745948753259; 9.997092322244717 28.283229206976976]
@test isapprox(pmean, 0.9754110613097936; atol=1e-10)
@test isapprox(B0, B2; atol=1e-10)

# L1 regularized fit
BL1    = fit(y, X, L1(0.5))
iris.p = [predict(BL1, X[i, :]) for i = 1:nrow(iris)]
pmean  = mean([iris.p[i][y[i]] for i = 1:nrow(iris)])
B2     = [1.4644363651728255e-14 -4.181402706775109; 2.9499498209148015e-15 -2.597332576788351; -2.5039102029980183 -5.407443881830369; 2.7955446590042112 6.689831401678983; 5.540820613266335e-15 5.815521890160558]
@test isapprox(pmean, 0.9297114521720717; atol=1e-10)
@test isapprox(BL1, B2; atol=1e-10)

# L2 regularized fit
BL2    = fit(y, X, L2(0.5))
iris.p = [predict(BL2, X[i, :]) for i = 1:nrow(iris)]
pmean  = mean([iris.p[i][y[i]] for i = 1:nrow(iris)])
B2     = [0.7888393490091326 -1.8380728113497073; -0.1999731251591173 -1.981765463406509; -1.821192471887587 -3.0788414903802224; 2.1840736652011308 4.507890037838459; -0.23844731040850425 3.4034321156498164]
@test isapprox(pmean, 0.8709932476955314; atol=1e-10)
@test isapprox(BL2, B2; atol=1e-10)

# Box regularized fit (constrain each parameter to be with [lowerbound, upperbound])
Bbox   = fit(y, X, BoxRegularizer(-5.0, 5.0))
iris.p = [predict(Bbox, X[i, :]) for i = 1:nrow(iris)]
pmean  = mean([iris.p[i][y[i]] for i = 1:nrow(iris)])
B2     = [-1.1542697077749056 -2.7633709517221177; 0.49175392755973313 -1.689148216224858; -2.465312842210372 -3.5685227186596347; 2.390464489277079 4.656881419478694; -1.338245423628683 3.0980999033071495]
@test isapprox(pmean, 0.8766776511261702; atol=1e-10)
@test isapprox(Bbox, B2; atol=1e-10)

end