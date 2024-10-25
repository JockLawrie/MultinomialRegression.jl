
@testset "ReadMe" begin

@info "$(now()) Starting test set: ReadMe"
iris  = dataset("datasets", "iris")
model = fit(@formula(Species ~ 1 + SepalWidth), iris)  # levels(iris.Species)[1] is the reference category
opts  = Dict(:iterations => 250, :g_abstol => 1e-8)    # Same terminology as Optim.Options
model = fit(@formula(Species ~ 1 + SepalWidth), iris; opts=opts)
Btrue = [18.858436607761437 12.99732439901066; -6.118961539097872 -4.079098097707158]
@test isapprox(coef(model), Btrue; atol=1e-8)

end
