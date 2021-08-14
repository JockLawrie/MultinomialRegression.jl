module fitpredict

export fit, predict

using LinearAlgebra
using Optim

using ..regularization

function fit(y, X, reg::Union{Nothing, AbstractRegularizer}=nothing, opts::Union{Nothing, Optim.Options}=nothing)
    nclasses = length(unique(y))
    nx       = size(X, 2)
    probs    = fill(0.0, nclasses)
    B0       = fill(0.0, nx * (nclasses - 1))
    fg!      = get_fg!(reg, probs, y, X)  # Debug with opts = Optim.Options(show_trace=true)
    mdl      = isnothing(opts) ? optimize(Optim.only_fg!(fg!), B0, LBFGS()) : optimize(Optim.only_fg!(fg!), B0, LBFGS(), opts)
    B        = reshape(mdl.minimizer, nx, nclasses - 1)
    reg isa BoxRegularizer && regularize!(B, reg)  # Constrain B to the box specified by reg
    B
end

predict(B, x) = update_probs!(fill(0.0, 1 + size(B, 2)), B, x)

################################################################################
# unexported

# L1 or L2 regularization
get_fg!(reg, probs, y, X) = (_, gradB, B) -> regularize(reg, B, gradB) - loglikelihood!(probs, y, X, B, gradB)

# No regularization
function get_fg!(reg::Nothing, probs, y, X)
    (_, gradB, B) -> begin
        fill!(gradB, 0.0)
        -loglikelihood!(probs, y, X, B, gradB)
    end
end

# Box regularization
function get_fg!(reg::BoxRegularizer, probs, y, X)
    nx       = size(X, 2)
    nclasses = length(probs)
    outB     = fill(0.0, nx * (nclasses - 1))
    (_, gradB, B) -> begin
        regularize!(outB, B, reg, gradB)
        -loglikelihood!(probs, y, X, outB, gradB)
    end
end

function loglikelihood!(probs, y, X, b::Vector, gradb::Vector)
    nx    = size(X, 2)
    nj    = Int(length(b) / nx)
    B     = reshape(b, nx, nj)
    gradB = reshape(gradb, nx, nj)
    loglikelihood!(probs, y, X, B, gradB)
end

function loglikelihood!(probs, y, X, B::Matrix, gradB::Matrix)
    LL = 0.0
    for (i, yi) in enumerate(y)
        x = view(X, i, :)
        update_probs!(probs, B, x)
        update_gradient!(gradB, probs, yi, x)
        LL += log(max(probs[yi], 1e-12))
    end
    LL
end

function update_probs!(probs, B, x)
    probs[1] = 0.0  # bx for first category is 0
    max_bx   = 0.0  # Find max bx for numerical stability
    nclasses = length(probs)
    for c = 2:nclasses
        bx       = dot(x, view(B, :, c - 1))
        probs[c] = bx
        max_bx   = bx > max_bx ? bx : max_bx
    end
    probs .= exp.(probs .- max_bx)  # Subtract max_bx first for numerical stability. Then prob[c] <= 1.
    normalize!(probs, 1)
end

function update_gradient!(gradB, probs, yi, x)
    nclasses = length(probs)
    nx = length(x)
    for c = 2:nclasses
        p = probs[c]
        for (j, xj) in enumerate(x)
            gradB[j, c - 1] += p * xj  # Negative gradient because function is to be minimized
        end
    end
    yi == 1 && return nothing
    for (j, xj) in enumerate(x)
        gradB[j, yi - 1] -= xj  # Negative gradient because function is to be minimized
    end
    nothing
end

end