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
    solver == :LBFGS && return fit_lbfgs(y, X, wts, reg, probs, opts)
    maxiter = isnothing(opts) ? 250  : get(opts, "maxiter", 250)
    tol     = isnothing(opts) ? 1e-9 : get(opts, "tol",     1e-9)
    fit_coordinatedescent(y, X, wts, probs, maxiter, tol)
end

"Returns either :LBFGS or :CoordinateDescent."
function select_solver(solver, reg, X)
    !isnothing(reg) && return :LBFGS  # Regularized models can't use IRLS/Coordinate Descent
    !isnothing(solver) && solver == :LBFGS && return :LBFGS  # If user specifies LBFGS then use it
    :CoordinateDescent
end

function fit_lbfgs(y, X, wts, reg, probs, opts)
    k    = size(probs, 2)
    n, p = size(X)
    Xtw  = isnothing(wts) ? nothing : fill(0.0, p, n)
    fg! = (_, g, b) -> begin
        B = reshape(b, p, k - 1)
        G = reshape(g, p, k - 1)
        update_probs!(probs, B, X)
        loss = -loglikelihood(y, probs, wts) + penalty(reg, b)
        gradient!(G, B, X, wts, reg, probs, y, Xtw)  # Modifies probs
        loss
    end
    b0     = fill(0.0, p * (k - 1))
    fitted = isnothing(opts) ? optimize(Optim.only_fg!(fg!), b0, LBFGS()) : optimize(Optim.only_fg!(fg!), b0, LBFGS(), opts)
    loss   = fitted.minimum
    B      = reshape(fitted.minimizer, p, k - 1)
    loss, B
end

function fit_coordinatedescent(y, X, wts, probs, maxiter, tol)
    k     = size(probs, 2)
    n, p  = size(X)
    B     = fill(0.0, p, k - 1)
    dB    = fill(0.0, p, k - 1)  # Bnew = B + dB
    W     = fill(0.0, n, k - 1)  # Working weights = wts .* p(1 - p)
    G     = fill(0.0, p, k - 1)  # G = gradient(-LL) = -gradient(LL) = Xt * (probs .- Y)
    Xt    = transpose(X)
    XtW   = fill(0.0, p, n)  # Xt * Diagonal(working weights)
    XtWX  = fill(0.0, p, p)  # Xt * Diagonal(working weights) * X
    probsview = view(probs, :, 2:k)
    converged = false
    loss_prev = Inf  # Required for the warning message if convergence is not achieved
    update_probs!(probs, B, X)
    loss = -loglikelihood(y, probs, wts)
    d = sqrt(eps())
    for iter = 1:maxiter
        # Update dB
        set_working_weights!(W, probsview, wts, d)  # Set working weights = W = wts .* p .* (1 .- p)
        probs_minus_y!(probs, y)
        gradient!(G, Xt, probsview, wts, XtW)  # gradient(-LL) = -gradient(LL)
        for j = 2:k
            multiply_3_matrices!(XtWX, XtW, Xt, Diagonal(view(W, :, j - 1)), X)
            ldiv!(view(dB, :, j - 1), cholesky!(Hermitian(XtWX)), view(G, :, j - 1))  # Or: ldiv!(dB_j, qr!(XtWX), G_j)
        end

        # Line search along dB
        a = 1.0
        loss_prev = loss
        for i_linesearch = 1:52  # eps() == 1 / (2^52)
            # Update B
            if i_linesearch == 1
                B .-= a .* dB  # The minus is due to the negative gradient used to obtain dB
            else
                B .+= a .* dB  # B = B + a_old*dB - a_new*dB = B + a_new*dB, since a_old = 2*a_new
            end

            # Update loss
            update_probs!(probs, B, X)
            loss = -loglikelihood(y, probs, wts)
            loss < loss_prev && break
            a *= 0.5  # Smaller step size
        end

        # Check for convergence
        converged = isapprox(loss, loss_prev; atol=tol) || iszero(loss_prev)
        converged && break
    end
    !converged && @warn "Coordinate Descent did not converge with tolerance $(tol). The last change in loss was $(abs(loss - loss_prev))"
    loss, B
end

"Given matrices A, B and C, compute the matrix product ABC, obtaining AB in the process."
function multiply_3_matrices!(ABC, AB, A, B, C)
    mul!(AB, A, B)
    mul!(ABC, AB, C)
end

function probs_minus_y!(probs, y)
    for (i, yi) in enumerate(y)
        probs[i, yi] -= 1.0
    end
end

"w .= wts .* Pi .* (1 - Pi)"
function set_working_weights!(w, probs, wts, d, i, j)
    Pi = view(probs, :, i)
    if i == j
        w .= wts .* max.(d, Pi .* (1.0 .- Pi))
    else
        Pj = view(probs, :, j)
        w .= wts .* min.(-d, -Pi .* Pj)
    end
end

function set_working_weights!(w, probs, wts::Nothing, d, i, j)
    Pi = view(probs, :, i)
    if i == j
        w .= max.(d, Pi .* (1.0 .- Pi))
    else
        Pj = view(probs, :, j)
        w .= min.(-d, -Pi .* Pj)
    end
end

function set_working_weights!(W, probs, wts, d)
    k = size(probs, 2)
    for j = 1:k
        set_working_weights!(view(W, :, j), probs, wts, d, j, j)
    end
end

################################################################################
# Loss, gradient, hessian

loglikelihood(y, probs, w)          = @inbounds sum(w[i]*log(max(probs[i, yi], 1e-12)) for (i, yi) in enumerate(y))
loglikelihood(y, probs, w::Nothing) = @inbounds sum(     log(max(probs[i, yi], 1e-12)) for (i, yi) in enumerate(y))

function update_probs!(probs, B, X)
    fill!(view(probs, :, 1), 0.0)
    mul!(view(probs, :, 2:size(probs, 2)), X, B)
    rowwise_softmax!(probs)
end

function rowwise_softmax!(eta::AbstractMatrix)
    ni = size(eta, 1)
    for i = 1:ni
        softmax!(view(eta, i, :))
    end
end

function softmax!(probs::AbstractVector)
    max_bx = -Inf
    for x in probs
        max_bx = x > max_bx ? x : max_bx
    end
    psum = 0.0
    @inbounds for (i, x) in enumerate(probs)
        probs[i] = exp(x - max_bx)
        psum += probs[i]
    end
    denom = 1.0/psum
    rmul!(probs, denom)
end

"grad(-LL + penalty)"
function gradient!(G, B, X, wts, reg, probs, y, Xtw)
    probs_minus_y!(probs, y)
    gradient!(G, transpose(X), view(probs, :, 2:size(probs, 2)), wts, Xtw)
    penalty_gradient!(G, reg, B)
end

# gradient(-LL) used in Coordinate Descent
gradient!(G, Xt, probs_minus_Y, wts::Nothing, Xtw) = mul!(G, Xt, probs_minus_Y)

# gradient(-LL) used in Coordinate Descent
function gradient!(G, Xt, probs_minus_Y, wts, Xtw)
    mul!(Xtw, Xt, Diagonal(wts))
    mul!(G, Xtw, probs_minus_Y)
end

"""
hessian(-LL + penalty)

Let k = the number of categories, and let p = the number of predictors.
The hessian is a (k-1) x (k-1) block matrix, with block size p x p.
In the code below, i and j denote the block indices; i.e., i and j each have k-1 values.
"""
function hessian!(H, X, wts, reg, probs, XtW, w)
    k  = size(probs, 2)
    p  = size(X, 2)
    Xt = transpose(X)
    d  = sqrt(eps())
    for j = 1:(k - 1)
        for i = j:(k - 1)
            rows = (p*(i - 1) + 1):(p*i)
            cols = (p*(j - 1) + 1):(p*j)
            set_working_weights!(w, probs, wts, d, i+1, j+1)
            multiply_3_matrices!(view(H, rows, cols), XtW, Xt, Diagonal(w), X)
        end
    end
    penalty_hessian!(H, reg)
    Hermitian(H, :L)
end

function hessian(X, wts, reg, probs)
    k    = size(probs, 2)
    n, p = size(X)
    H    = fill(0.0, p*(k - 1), p*(k - 1))
    XtW  = fill(0.0, p, n)
    w    = fill(0.0, n)
    hessian!(H, X, wts, reg, probs, XtW, w)
end

end