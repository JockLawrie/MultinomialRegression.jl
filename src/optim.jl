module optim

export fit_optim

using LinearAlgebra
using Logging
using Optim

using ..regularization

function fit_optim(y, X, wts::Union{Nothing, AbstractVector}=nothing, reg::Union{Nothing, AbstractRegularizer}=nothing,
                   solver=nothing, opts::Union{Nothing, T}=nothing) where{T}
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
    solver = select_solver(solver, reg, X)
    Y      = construct_Y(y, size(probs)...)
    solver == :LBFGS && return fit_lbfgs(y, X, wts, reg, probs, Y, opts)
    maxiter = isnothing(opts) ? 250  : get(opts, "maxiter", 250)
    tol     = isnothing(opts) ? 1e-6 : get(opts, "tol",     1e-6)
    fit_irls(y, X, wts, probs, Y, maxiter, tol)
end

"Returns either :LBFGS or :IRLS."
function select_solver(solver, reg, X)
    !isnothing(reg) && return :LBFGS  # Regularized models can't use IRLS
    !isnothing(solver) && solver == :LBFGS && return :LBFGS  # If user specifies LBFGS then use it
    :IRLS
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
        B = reshape(b, p, k - 1)
        G = reshape(g, p, k - 1)
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

function fit_irls(y, X, wts, probs, Y, maxiter, tol)
    n, p  = size(X)
    k     = size(Y, 2)
    B     = fill(0.0, p, k)
    eta   = fill(0.0, n, k)   # eta = XB
    p_1mp = fill(0.0, n, k)   # p(1 - p)
    Xtw   = fill(0.0, p, n)   # Xt * Diagonal(working weights)
    XtwX  = fill(0.0, p, p)   # Xt * Diagonal(working weights) * X
    XtwResid  = fill(0.0, p)  # Xtw * eta_j = Xt * Diagonal(working weights) * eta[:, j]
    Xt        = transpose(X)
    converged = false
    loss_prev = Inf
    mul!(eta, X, B)
    copyto!(probs, eta)
    rowwise_softmax!(probs)
    loss = -loglikelihood(y, probs, wts)
    for iter = 1:maxiter
        # Update B
        p_1mp .= max.(probs .* (1.0 .- probs), sqrt(eps()))
        eta  .+= (Y .- probs) ./ p_1mp    # Set eta   = working residuals = X*B .+ (Y -. probs) ./ p_1mp
        set_working_weights!(p_1mp, wts)  # Set p_1mp = working weights   = wts .* p_1mp
        for j = 2:k
            w     = view(p_1mp, :, j)  # Working weights
            resid = view(eta, :, j)    # Working residuals
            B_j   = view(B, :, j)
            mul!(Xtw,  Xt, Diagonal(w))
            mul!(XtwX, Xtw, X)
            mul!(XtwResid, Xtw, resid)
            #ldiv!(B_j, qr!(XtwX), XtwResid)
            #ldiv!(B_j, cholesky!(Hermitian(XtwX)), XtwResid)
            C = cholesky!(Hermitian(XtwX)).factors
            ldiv!(B_j, LowerTriangular(transpose(C)), XtwResid)
            ldiv!(B_j, UpperTriangular(C), B_j)
        end
        # Update loss
        loss_prev = loss
        mul!(eta, X, B)
        copyto!(probs, eta)
        rowwise_softmax!(probs)
        loss = -loglikelihood(y, probs, wts)
        converged = isapprox(loss, loss_prev; rtol=tol) || iszero(loss_prev)
        converged && break
    end
    !converged && @warn "IRLS did not converge with tolerance $(tol). The last change in loss was $(abs(loss - loss_prev))"
    loss, Matrix(B[:, 2:k])
end

set_working_weights!(p_1mp, wts::Nothing) = nothing
set_working_weights!(p_1mp, wts) = (p_1mp .*= wts)

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