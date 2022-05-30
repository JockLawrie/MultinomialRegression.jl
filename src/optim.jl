module optim

export fit_optim

using LinearAlgebra
using Logging
using Optim

using ..regularization

function fit_optim(y, X, wts::Union{Nothing, AbstractVector}=nothing, reg::Union{Nothing, AbstractRegularizer}=nothing,
                   solver=nothing, opts::Union{Nothing, Optim.Options}=nothing)
    nclasses = length(unique(y))
    nobs     = size(X, 1)
    probs    = fill(0.0, nobs, nclasses)
    loss, B  = optimise(y, X, wts, reg, solver, opts, probs)
    update_probs!(probs, B, X)
    H = hessian(X, wts, nothing, probs)  # hessian(-LL), not hessian(-LL + penalty)
    if rank(H) == length(B)
        if !isnothing(reg)
            @warn "Regularisation implies that the covariance matrix and standard errors are not estimated from MLEs"
        end
        vcov = Matrix(Hermitian(inv(bunchkaufman!(H))))  # varcov(b) = inv(FisherInformation) = inv(Hessian(-LL))
    else
        @warn "Standard errors cannot be computed (Hessian does not have full rank). Check for linearly dependent predictors."
        vcov = fill(NaN, length(B), length(B))
    end
    loss, B, vcov
end

################################################################################
# Unexported

function optimise(y, X, wts, reg, solver, opts, probs)
    solver = select_solver(solver, X)
    Y      = construct_Y(y, size(probs)...)
    if solver == :LBFGS
        return fit_lbfgs(y, X, wts, reg, probs, Y, opts)
    elseif solver == :Newton
        maxiter = isnothing(opts) ? 1000 : opts["maxiter"]
        tol     = isnothing(opts) ? 1e-9 : opts["tol"]
        if isnothing(reg)
            return fit_irls(y, X, wts, probs, Y, maxiter, tol)  # Special case of Newton that is faster
        else
            return fit_newton(y, X, wts, reg, probs, Y, maxiter, tol)
        end
    else
        error("Unrecognised solver: $(solver)")
    end
end

function select_solver(solver, X)
    result = isnothing(solver) ? :Newton : solver  # Default to Newton
    if result == :Newton
        c = cond(transpose(X) * X)
        if c >= 30.0
            @warn "X contains linear dependencies (condition number is $(c)). Switching to the LBFGS solver."
            result = :LBFGS
        end
    end
    result
end

function construct_Y(y, nobs, nclasses)
    Y = fill(0.0, nobs, nclasses)
    for (i, yi) in enumerate(y)
        Y[i, yi] = 1.0
    end
    Y
end

function fit_lbfgs(y, X, wts, reg, probs, Y, opts)
    p   = size(X, 2)
    k   = size(Y, 2)
    fg! = (_, g, b) -> begin
        B  = reshape(b, p, k - 1)
        G  = reshape(g, p, k - 1)
        update_probs!(probs, B, X)
        gradient!(G, B, X, wts, reg, probs, Y)
        -loglikelihood(y, probs, wts) + penalty(reg, b)
    end
    b0     = fill(0.0, p * (k - 1))
    fitted = isnothing(opts) ? optimize(Optim.only_fg!(fg!), b0, LBFGS()) : optimize(Optim.only_fg!(fg!), b0, LBFGS(), opts)
    loss   = fitted.minimum
    B      = reshape(fitted.minimizer, p, k - 1)
    loss, B
end

function fit_newton(y, X, wts, reg, probs, Y, maxiter, tol)
#=
    p    = size(X, 2)
    k    = size(Y, 2)
    fgh! = (_, g, H, b) -> begin
        B = reshape(b, p, k - 1)
        G = reshape(g, p, k - 1)
        update_probs!(probs, B, X)
        gradient!(G, B, X, wts, reg, probs, Y)  # grad(-LL + penalty)
        H2 = hessian!(H, X, wts, reg, probs)    # hessian(-LL + penalty). H2 is Hermitian
        copyto!(H, H2)
        -loglikelihood(y, probs, wts) + penalty(reg, b)
    end
    b0     = fill(0.0, p * (k - 1))
    fitted = isnothing(opts) ? optimize(Optim.only_fgh!(fgh!), b0, Newton()) : optimize(Optim.only_fgh!(fgh!), b0, Newton(), opts)
    loss   = fitted.minimum
    B      = reshape(fitted.minimizer, p, k - 1)
    return loss, B
=#
    n, p = size(X)
    k    = size(Y, 2)
    B    = fill(0.0, p, k - 1)
    G    = fill(0.0, p, k - 1)
    H    = fill(0.0, p*(k - 1), p*(k - 1))
    b    = reshape(B, length(B))
    g    = reshape(G, length(G))
    loss      = Inf
    loss_prev = Inf
    converged = false
    for iter = 1:maxiter
        loss_prev = loss
        update_probs!(probs, B, X)
        loss = -loglikelihood(y, probs, wts) + penalty(reg, B)
        gradient!(G, B, X, wts, reg, probs, Y)  # G = gradient(-LL + penalty)
        H2 = hessian!(H, X, wts, reg, probs)    # H = hessian(-LL + penalty), H2 = Hermitian(H)
        cholesky!(H2, Val(false))
        b .-= H2\g
        converged = isapprox(loss, loss_prev; atol=tol) || iszero(loss_prev)
        converged && break
    end
    !converged && @warn "The Newton solver did not converge with tolerance $(tol). The last change in loss was $(abs(loss - loss_prev))"
    loss, B
end

@views function fit_irls(y, X, wts, probs, Y, maxiter, tol)
    n, p  = size(X)
    k     = size(Y, 2)
    Xt    = transpose(X)
    B     = fill(0.0, p, k)
    eta   = fill(0.0, n, k)
    p_1mp = fill(0.0, n, k)
    loss       = Inf
    loss_prev  = Inf
    converged  = false
    for iter = 1:maxiter
        loss_prev = loss
        mul!(eta, X, B)
        copyto!(probs, eta)
        rowwise_softmax!(probs)
        loss   = -loglikelihood(y, probs, wts)
        p_1mp .= max.(probs .* (1.0 .- probs), sqrt(eps()))
        eta  .+= (Y .- probs) ./ p_1mp
        update_p_1mp!(p_1mp, wts)  # p_1mp[i, :] = wts[i] .* p_1mp[i, :] for each i
        for j = 2:k
            C       = cholesky!(Hermitian(Xt * Diagonal(p_1mp[:, j]) * X)).factors
            B[:, j] = LowerTriangular(transpose(C)) \ (Xt * Diagonal(p_1mp[:, j]) * eta[:, j])
            B[:, j] = UpperTriangular(C) \ B[:, j]
        end
        converged = isapprox(loss, loss_prev; atol=tol) || iszero(loss_prev)
        converged && break
    end
    !converged && @warn "IRLS did not converge with tolerance $(tol). The last change in loss was $(abs(loss - loss_prev))"
    loss, Matrix(B[:, 2:k])
end

update_p_1mp!(p_1mp, wts::Nothing) = nothing
update_p_1mp!(p_1mp, wts) = (p_1mp .*= wts)

loglikelihood(y, probs, w)          = @inbounds sum(w[i]*log(max(probs[i, yi], 1e-12)) for (i, yi) in enumerate(y))
loglikelihood(y, probs, w::Nothing) = @inbounds sum(     log(max(probs[i, yi], 1e-12)) for (i, yi) in enumerate(y))

function update_probs!(probs, B, X)
    probs[:, 1]     .= 0.0
    probs[:, 2:end] .= X * B  # eta
    rowwise_softmax!(probs)
end

function rowwise_softmax!(eta::AbstractMatrix)
    ni = size(eta, 1)
    for i = 1:ni
        softmax!(view(eta, i, :))
    end
end

function softmax!(probs::AbstractVector)
    max_bx = maximum(probs)
    probs .= exp.(probs .- max_bx)  # Subtract max_bx first for numerical stability. Then prob[c] <= 1.
    normalize!(probs, 1)
end

"grad(-LL + penalty)"
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
hessian(-LL + penalty)

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
                H[rows, cols] = Xt * Diagonal(probs[:, i + 1] .* (1 .- probs[:, i + 1]) .* wts) * X
            else
                H[rows, cols] = Xt * Diagonal(-probs[:, i + 1] .* probs[:, j + 1] .* wts) * X
            end
        end
    end
    penalty_hessian!(H, reg)
    Hermitian(H, :L)
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
                H[rows, cols] = Xt * Diagonal(probs[:, i + 1] .* (1 .- probs[:, i + 1])) * X
            else
                H[rows, cols] = Xt * Diagonal(-probs[:, i + 1] .* probs[:, j + 1]) * X
            end
        end
    end
    penalty_hessian!(H, reg)
    Hermitian(H, :L)
end

function hessian(X, wts, reg, probs)
    k = size(probs, 2)  # nclasses
    p = size(X, 2)      # npredictors
    H = fill(0.0, p*(k - 1), p*(k - 1))
    hessian!(H, X, wts, reg, probs)
end

end