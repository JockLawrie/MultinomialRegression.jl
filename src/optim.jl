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
    H = hessian(X, wts, reg, probs)
    if rank(H) == length(B)
        if penalty(reg, B) == 0.0
            @warn "Regularisation implies that the covariance matrix and standard errors are not estimated from MLEs"
        end
        vcov = Matrix(Hermitian(inv(bunchkaufman!(-H))))  # varcov(b) = inv(FisherInformation) = inv(-Hessian)
    else
        @warn "Standard errors cannot be computed (Hessian does not have full rank). Check for linearly dependent predictors."
        vcov = fill(NaN, length(B), length(B))
    end
    loss, B, vcov
end

################################################################################
# Unexported

function optimise(y, X, wts, reg, solver, opts, probs)
    solver   = isnothing(solver) ? :LBFGS : solver
    nclasses = length(unique(y))
    nobs, nx = size(X)
    Y        = construct_Y(y, nobs, nclasses)
    if solver == :LBFGS
        fg!   = get_fg!(y, X, wts, reg, probs, Y)
        b0    = fill(0.0, nx * (nclasses - 1))
        model = isnothing(opts) ? optimize(Optim.only_fg!(fg!), b0, LBFGS()) : optimize(Optim.only_fg!(fg!), b0, LBFGS(), opts)
        loss  = model.minimum
        B     = reshape(model.minimizer, nx, nclasses - 1)
        return loss, B
    elseif solver == :Newton
        fgh!  = get_fgh!(y, X, wts, reg, probs, Y)
        b0    = fill(0.0, nx * (nclasses - 1))
        model =  isnothing(opts) ? optimize(Optim.only_fgh!(fgh!), b0, Newton()) : optimize(Optim.only_fgh!(fgh!), b0, Newton(), opts)
        loss  = model.minimum
        B     = reshape(model.minimizer, nx, nclasses - 1)
        return loss, B
    elseif solver == :IRLS
        maxiter = isnothing(opts) ? 250  : opts["maxiter"]
        tol     = isnothing(opts) ? 1e-9 : opts["tol"]
        loss, B = fit_irls(y, X, wts, reg, probs, Y, maxiter, tol)
        return loss, B
    else
        error("Unrecognised solver: $(solver)")
    end
end

function construct_Y(y, nobs, nclasses)
    Y = fill(0.0, nobs, nclasses)
    for (i, yi) in enumerate(y)
        Y[i, yi] = 1.0
    end
    Y
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

@views function fit_irls(y, X, wts, reg, probs, Y, maxiter, tol)
    #F, lin_ind = omit_collinear_variables(X)
    F     = qr(X, ColumnNorm())
    Q     = Matrix(F.Q)
    Qt    = transpose(Q)
    n, p  = size(F)
    k     = size(Y, 2)
    B     = fill(0.0, p, k)
    eta   = fill(0.0, n, k)
    p_1mp = fill(0.0, n, k)
    loss       = Inf
    loss_prev  = Inf
    converged  = false
    for iter = 1:maxiter
        loss_prev = loss
        mul!(eta, Q, B)
        copyto!(probs, eta)
        rowwise_softmax!(probs)
        loss   = -loglikelihood(y, probs, wts) + penalty(reg, B)
        p_1mp .= max.(probs .* (1.0 .- probs), sqrt(eps()))
        eta  .+= (Y .- probs) ./ p_1mp
        update_p_1mp!(p_1mp, wts)  # p_1mp[i, :] = wts[i] .* p_1mp[i, :] for each i
        for j = 2:k
            C       = cholesky!(Hermitian(Qt * Diagonal(p_1mp[:, j]) * Q)).factors
            B[:, j] = LowerTriangular(transpose(C)) \ (Qt * Diagonal(p_1mp[:, j]) * eta[:, j])
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

function omit_collinear_variables(X)
    F   = qr(X, Val(true))
    qrr = count(x -> abs(x) ≥ sqrt(eps()), diag(F.R))
    if qrr < size(F, 2)
        lin_ind = sort!(invperm(F.p)[1:qrr])
        X2 = convert(Matrix{Float64}, X[:,lin_ind])
        F  = qr(X2, Val(true))
    else
        lin_ind = collect(1:size(F, 2))
    end
    F, lin_ind
end

end