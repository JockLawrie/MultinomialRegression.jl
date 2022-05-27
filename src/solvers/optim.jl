module lbfgs

export fit_optim

using LinearAlgebra
using Logging
using Optim

using ..regularization

function fit_optim(y, X, wts::Union{Nothing, AbstractVector}=nothing, reg::Union{Nothing, AbstractRegularizer}=nothing,
                   solver=nothing, opts::Union{Nothing, Optim.Options}=nothing)
    nclasses = length(unique(y))
    nobs, nx = size(X)
    probs = fill(0.0, nobs, nclasses)
    model = optimise(y, X, wts, reg, solver, opts, probs)
    b     = model.minimizer
    B     = reshape(b, nx, nclasses - 1)
    loss  = model.minimum
    LL    = penalty(reg, b) - loss  # loss = -LL + penalty
    update_probs!(probs, B, X)
    H = hessian(X, wts, reg, probs)
    if rank(H) == length(b)
        if penalty(reg, b) == 0.0
            @warn "Regularisation implies that the covariance matrix and standard errors are not estimated from MLEs"
        end
        vcov = Matrix(Hermitian(inv(bunchkaufman!(-H))))  # varcov(b) = inv(FisherInformation) = inv(-Hessian)
    else
        @warn "Standard errors cannot be computed (Hessian does not have full rank). Check for linearly dependent predictors."
        vcov = fill(NaN, length(b), length(b))
    end
    loss, B, vcov
end

################################################################################
# Unexported

function construct_Y(y, nobs, nclasses)
    Y = fill(0.0, nobs, nclasses)
    for (i, yi) in enumerate(y)
        Y[i, yi] = 1.0
    end
    Y
end

function optimise(y, X, wts, reg, solver, opts, probs)
    solver   = isnothing(solver) ? :Newton : solver
    nclasses = length(unique(y))
    nobs, nx = size(X)
    b0       = fill(0.0, nx * (nclasses - 1))
    if solver == :NelderMead
        f = get_f(y, X, wts, reg, probs)
        return isnothing(opts) ? optimize(f, b0, NelderMead()) : optimize(f, b0, NelderMead(), opts)
    elseif solver == :LBFGS
        Y   = construct_Y(y, nobs, nclasses)
        fg! = get_fg!(y, X, wts, reg, probs, Y)
        return isnothing(opts) ? optimize(Optim.only_fg!(fg!), b0, LBFGS()) : optimize(Optim.only_fg!(fg!), b0, LBFGS(), opts)
    elseif solver == :Newton
        Y    = construct_Y(y, nobs, nclasses)
        fgh! = get_fgh!(y, X, wts, reg, probs, Y)
        return isnothing(opts) ? optimize(Optim.only_fgh!(fgh!), b0, Newton()) : optimize(Optim.only_fgh!(fgh!), b0, Newton(), opts)
    else
        error("Unrecognised solver: $(solver)")
    end
end

function get_f(y, X, wts, reg, probs)
    (b) -> begin
        ni, nx = size(X)
        nj = Int(length(b) / nx)  # nclasses - 1
        B  = reshape(b, nx, nj)
        update_probs!(probs, B, X)
        -loglikelihood(y, probs, wts) + penalty(reg, b)
    end
end

function get_fg!(y, X, wts, reg, probs, Y)
    (_, g, b) -> begin
        ni, nx = size(X)
        nj = Int(length(b) / nx)  # nclasses - 1
        B  = reshape(b, nx, nj)
        G  = reshape(g, nx, nj)
        update_probs!(probs, B, X)
        gradient!(G, B, X, wts, reg, probs, Y)
        -loglikelihood(y, probs, wts) + penalty(reg, b)
    end
end

function get_fgh!(y, X, wts, reg, probs, Y)
    (_, g, H, b) -> begin
        ni, nx = size(X)
        nj = Int(length(b) / nx)  # nclasses - 1
        B  = reshape(b, nx, nj)
        G  = reshape(g, nx, nj)
        update_probs!(probs, B, X)
        gradient!(G, B, X, wts, reg, probs, Y)
        hessian!(H, X, wts, reg, probs)
        -loglikelihood(y, probs, wts) + penalty(reg, b)
    end
end

loglikelihood(y, probs, w)          = @inbounds sum(w[i]*log(max(probs[i, yi], 1e-12)) for (i, yi) in enumerate(y))
loglikelihood(y, probs, w::Nothing) = @inbounds sum(     log(max(probs[i, yi], 1e-12)) for (i, yi) in enumerate(y))

function update_probs!(probs, B, X)
    ni = size(X, 1)
    probs[:, 1]     .= 0.0
    probs[:, 2:end] .= X * B
    for i = 1:ni
        eta_to_probs!(view(probs, i, :))
    end
end

function eta_to_probs!(probs::AbstractVector)
    max_bx = 0.0  # Find max bx for numerical stability
    for (i, bx) in enumerate(probs)
        max_bx = bx > max_bx ? bx : max_bx
    end
    probs .= exp.(probs .- max_bx)  # Subtract max_bx first for numerical stability. Then prob[c] <= 1.
    normalize!(probs, 1)
end

function gradient!(G, B, X, wts, reg, probs, Y)
    nclasses = size(Y, 2)
    Y2 = view(Y, :, 2:nclasses)
    P2 = view(probs, :, 2:nclasses)
    G .= transpose(X) * Diagonal(wts) * (P2 .- Y2)  # Negative gradient because -LL is to be minimized
    penalty_gradient!(G, reg, B)
end

function gradient!(G, B, X, wts::Nothing, reg, probs, Y)
    nclasses = size(Y, 2)
    Y2 = view(Y, :, 2:nclasses)
    P2 = view(probs, :, 2:nclasses)
    G .= transpose(X) * (P2 .- Y2)  # Negative gradient because -LL is to be minimized
    penalty_gradient!(G, reg, B)
end

"""
Let k = the number of categories, and let p = the number of predictors.
The hessian is a (k-1) x (k-1) block matrix, with block size p x p.
In the code below, i and j denote the block indices; i.e., i and j each have k-1 values.
"""
@views function hessian!(H, X, wts, reg, probs)
    k  = size(probs, 2)  # nclasses
    p  = size(X, 2)      # npredictors
    Xt = transpose(X)
    for j = 1:(k - 1)
        for i = j:(k - 1)
            rows = (p*(i - 1) + 1):(p*i)
            cols = (p*(j - 1) + 1):(p*j)
            if i == j
                H[rows, cols] = Xt * Diagonal(probs[:, i + 1] .* (probs[:, i + 1] .- 1) .* wts) * X
            else
                H[rows, cols] = Xt * Diagonal(probs[:, i + 1] .* probs[:, j + 1] .* wts) * X
            end
        end
    end
    H2 = Hermitian(H, :L)
    penalty_hessian!(H2, reg)
    copyto!(H, H2)
end

@views function hessian!(H, X, wts::Nothing, reg, probs)
    k  = size(probs, 2)  # nclasses
    p  = size(X, 2)      # npredictors
    Xt = transpose(X)
    for j = 1:(k - 1)
        for i = j:(k - 1)
            rows = (p*(i - 1) + 1):(p*i)
            cols = (p*(j - 1) + 1):(p*j)
            if i == j
                H[rows, cols] = Xt * Diagonal(probs[:, i + 1] .* (probs[:, i + 1] .- 1)) * X
            else
                H[rows, cols] = Xt * Diagonal(probs[:, i + 1] .* probs[:, j + 1]) * X
            end
        end
    end
    H2 = Hermitian(H, :L)
    penalty_hessian!(H2, reg)
    copyto!(H, H2)
end

function hessian(X, wts, reg, probs)
    k = size(probs, 2)  # nclasses
    p = size(X, 2)      # npredictors
    H = fill(0.0, p*(k - 1), p*(k - 1))
    hessian!(H, X, wts, reg, probs)
    H
end

end