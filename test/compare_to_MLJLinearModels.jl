@testset "Compare to MLJLinearModels.jl"  begin

@info "$(now()) Starting test set: Compare to MLJLinearModels.jl"

################################################################################
# This code is from MLJLinearModels.jl/test/testutils.jl

sparsify!(θ, s, r) = (θ .*= (rand(r, length(θ)) .< s))

function multi_rand(Mp, r)
    # Mp[i, :] sums to 1
    n, c = size(Mp)
    be   = reshape(rand(r, length(Mp)), n, c)
    y    = zeros(Int, n)
    @inbounds for i in eachindex(y)
        rp = 1.0
        for k in 1:c-1
            if (be[i, k] < Mp[i, k] / rp)
                y[i] = k
                break
            end
            rp -= Mp[i, k]
        end
    end
    y[y .== 0] .= c
    return y
end

function generate_multiclass(n, p, c; seed=53412224, sparse=1)
    r   = StableRNG(seed)
    X   = randn(r, n, p)
    X1  = MLJLinearModels.augment_X(X, true)
    θ   = randn(r, p * c)
    θ1  = randn(r, (p+1) * c)
    sparse < 1 && begin
        sparsify!(θ, sparse, r)
        sparsify!(θ1, sparse, r)
    end
    y   = zeros(Int, n)
    y1  = zeros(Int, n)

    P   = MLJLinearModels.apply_X(X, θ, c)
    M   = exp.(P)
    Mn  = M ./ sum(M, dims=2)
    P1  = MLJLinearModels.apply_X(X, θ1, c)
    M1  = exp.(P1)
    Mn1 = M1 ./ sum(M1, dims=2)

    y  = multi_rand(Mn, r)
    y1 = multi_rand(Mn1, r)
    return ((X, y, θ), (X1, y1, θ1))
end

################################################################################
# Generate data
n = 1000
p = 5
c = 3
((X0, y0, theta0), (X1, y1, theta1)) = generate_multiclass(n, p, c)  # Data without/with intercept

# Split into train/test data
ntrain = floor(Int, 0.8 * n)
ntest  = n - ntrain
rp     = randperm(n)
itrain = rp[1:ntrain]
itest  = rp[(ntrain + 1):n]
ytrain, ytest = y1[itrain], y1[itest]
Xtrain, Xtest = X1[itrain, :], X1[itest, :]

################################################################################
# Unregularized fit

# MultinomialRegression
B  = fit(ytrain, Xtrain).params
p  = [predict(B, Xtest[i, :]) for i = 1:ntest]
LL = sum(log.([p[i][ytest[i]] for i = 1:ntest]))

# MLJLinearModels
B_mlj  = mlj_fit(ytrain, Xtrain; fit_intercept=false, penalty=:none)
p_mlj  = mlj_predict(Xtest, B_mlj)
LL_mlj = sum(log.([p_mlj[i, ytest[i]] for i = 1:ntest]))
@test isapprox(LL, LL_mlj; rtol=0.01)

################################################################################
# L1 fit

# MultinomialRegression
B  = fit(ytrain, Xtrain, L1(0.5)).params
p  = [predict(B, Xtest[i, :]) for i = 1:ntest]
LL = sum(log.([p[i][ytest[i]] for i = 1:ntest]))

# MLJLinearModels
B_mlj  = mlj_fit(ytrain, Xtrain; fit_intercept=false, gamma=0.5)
p_mlj  = mlj_predict(Xtest, B_mlj)
LL_mlj = sum(log.([p_mlj[i, ytest[i]] for i = 1:ntest]))
@test isapprox(LL, LL_mlj; rtol=0.01)

################################################################################
# L2 fit

# MultinomialRegression
B  = fit(ytrain, Xtrain, L2(0.5)).params
p  = [predict(B, Xtest[i, :]) for i = 1:ntest]
LL = sum(log.([p[i][ytest[i]] for i = 1:ntest]))

# MLJLinearModels
B_mlj  = mlj_fit(ytrain, Xtrain; fit_intercept=false, lambda=0.5)
p_mlj  = mlj_predict(Xtest, B_mlj)
LL_mlj = sum(log.([p_mlj[i, ytest[i]] for i = 1:ntest]))
@test isapprox(LL, LL_mlj; rtol=0.01)

end