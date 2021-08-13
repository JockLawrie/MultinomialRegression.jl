module fitpredict

export fit, predict

using LinearAlgebra
using Optim

using ..regularization

function fit(y, X, reg::Union{Nothing, AbstractRegularizer}=nothing)
    ycount   = countmap(y)
    nclasses = length(ycount)
    probs    = fill(0.0, nclasses)
    nx       = size(X, 2)
    loss     = get_loss_function(reg, probs, y, X, nx, nclasses)
    B0       = fill(0.0, nx * (nclasses - 1))
    opts     = Optim.Options(time_limit=60, f_tol=1e-6)  # Debug with show_trace=true
    mdl      = optimize(loss, B0, LBFGS(), opts)
    B        = reshape(mdl.minimizer, nx, nclasses - 1)
    reg isa BoxRegularizer && regularize!(B, B, reg)  # Constrain B to the box specified by reg
    B
end

predict(B, x) = update_probs!(fill(0.0, 1 + size(B, 2)), B, x)

################################################################################
# unexported

get_loss_function(reg, probs, y, X, nx, nclasses) = B -> regularize(reg, B) - loglikelihood!(probs, y, X, B)

get_loss_function(reg::Nothing, probs, y, X, nx, nclasses) = B -> -loglikelihood!(probs, y, X, B)

function get_loss_function(reg::BoxRegularizer, probs, y, X, nx, nclasses)
    outB = fill(0.0, nx, nclasses - 1)
    B -> begin
        regularize!(outB, B, reg)
        -loglikelihood!(probs, y, X, outB)
    end
end

function loglikelihood!(probs, y, X, B)
    LL = 0.0
    nx = size(X, 2)
    nj = Int(length(B) / nx)
    B2 = reshape(B, nx, nj)
    ni = length(y)
    for i = 1:ni
        update_probs!(probs, B2, view(X, i, :))
        p   = max(probs[y[i]], 1e-10)
        LL += log(p)
    end
    LL
end

function update_probs!(probs, B, x)
    # Populate probs with the bx and find their maximum
    probs[1] = 0.0  # bx for first category is 0
    max_bx   = 0.0
    nclasses = length(probs)
    for j = 2:nclasses
        bx       = dot(x, view(B, :, j - 1))
        probs[j] = bx
        max_bx   = bx > max_bx ? bx : max_bx
    end
    # Calculate the numerators and the denominator of the probabilities
    denom = 0.0
    for j = 1:nclasses
        probs[j] = max(exp(probs[j] - max_bx), 1e-12)  # Subtract max_bx first for numerical stability (then val <= 1)
        denom   += probs[j]
    end
    probs ./= denom  # Normalise. We know that denom >= 1, so this will work.
end

function countmap(y)
    result = Dict{eltype(y), Int}()
    for x in y
        if haskey(result, x)
            result[x] += 1
        else
            result[x] = 1
        end
    end
    result
end

end