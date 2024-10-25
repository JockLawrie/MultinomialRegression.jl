module optim

export fit_optim

using LinearAlgebra
using Logging

using ..regularization

function fit_optim(y, X, wts::Union{Nothing, AbstractVector}=nothing,
                   reg::Union{Nothing, AbstractRegularizer}=nothing,
                   opts::Union{Nothing, T}=nothing) where{T}
    nclasses = length(unique(y))
    nobs     = size(X, 1)
    probs    = fill(0.0, nobs, nclasses)
    opts     = override_default_options(defaultopts(), opts)  # User-specified options override default options
    loss, B  = blockwise_coordinatedescent(y, X, wts, reg, probs, opts[:iterations], opts[:g_abstol])
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

defaultopts() = Dict(:iterations => 250, :f_abstol => 1e-9, :g_abstol => 1e-8)

"Override default options with user-specified options"
override_default_options(default_opts, opts::Nothing) = default_opts

function override_default_options(default_opts, opts)
    isempty(opts) && return default_opts
    opts2 = Dict{Symbol, valtype(opts)}(Symbol(k) => v for (k, v) in opts)
    merge!(default_opts, opts2)
end

################################################################################
# Solve with Coordinate Descent

"""
Set the search direction dB using the Newton-Raphson method.
That is, set dB = inv(H)*g.
Could replace qr!(H) with cholesky!(Hermitian(H)); the former is more stable but slower.
"""
set_searchdirection_newtonraphson!(dB, H, g) = ldiv!(dB, qr!(H), g)

"Block-wise cyclic coordinate descent"
function blockwise_coordinatedescent(y, X, wts, reg, probs, iterations, g_abstol)
    km1  = size(probs, 2) - 1
    n, p = size(X)
    B    = fill(0.0, p, km1)
    dB   = fill(0.0, p, km1)  # Bnew = B + dB
    g    = fill(0.0, p)       # 1 column of G = gradient(-LL) = -gradient(LL) = Xt * diag(wts) * (probs .- Y)
    H    = fill(0.0, p, p)    # A block on the diagonal of the Hessian. Xt * Diagonal(ww) * X
    ww   = fill(0.0, n)       # 1 column of working weights = wts .* p .* (1 .- p)
    Xtw  = isnothing(wts) ? transpose(X) : transpose(X)*Diagonal(wts)
    XtW  = fill(0.0, p, n)  # Xt * Diagonal(ww)
    loss = loss!(y, X, wts, reg, B, probs)
    d    = sqrt(eps())
    converged = false
    XtwY = construct_XtwY(Xtw, y, km1)  # Xt*diag(wts)*Y (because G = Xtw*probs .- XtwY)BackTracking()
    for iter = 1:iterations
        for j = 1:km1
            B_j  = view(B, :, j)
            dB_j = view(dB, :, j)
            gradient_group!(g, j, y, X, wts, reg, B, probs, Xtw, XtwY)
            hessian_group!(H, j, X, wts, reg, probs, XtW, ww, d)
            set_searchdirection_newtonraphson!(dB_j, H, g)
            loss = linesearch!(B_j, dB_j, y, X, wts, reg, B, probs, loss)
        end
        converged = isapprox(norm(dB, Inf), 0.0; atol=g_abstol)
        converged && break
    end
    !converged && @warn "Blockwise Coordinate Descent did not converge: gnorm = $(norm(dB, Inf)) > g_abstol ($(g_abstol))."
    loss, B
end

"""
1. Start with a large step size, a = 2, and a step size multiplier m in (0, 1).
2. Set B = B - a*dB
3. If loss(B, ...) < prevloss, return loss, else a *= m and go back to Step 2.
"""
function linesearch!(B_j, dB_j, y, X, wts, reg, B, probs, loss_prev)
    loss  = Inf
    aprev = 0.0  # Previous step size
    a     = 2.0  # Initial step size
    m     = 0.8  # At each iteration, multiply the step size by m
    while a > 1e-7
        # Update B
        c = aprev - a
        B_j .+= c .* dB_j  # B = B + a_old*dB - a_new*dB = B + (a_old - a_new)*dB = B + c*dB

        # Update loss
        loss = loss!(y, X, wts, reg, B, probs)
        loss < loss_prev && break
        aprev = a
        a    *= m  # Smaller step size
    end
    loss
end

################################################################################
# Loss function

"Modifies (updates) probs"
function loss!(y, X, wts, reg, B, probs)
    update_probs!(probs, B, X)
    -loglikelihood(y, probs, wts) + penalty(reg, B)
end

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

################################################################################
# Gradient

"Gradient for j^th group of parameters"
function gradient_group!(g, j, y, X, wts, reg, B, probs, Xtw, XtwY)
    b  = view(B, :, j)
    c  = view(XtwY, :, j)
    ps = view(probs, :, j+1)
    mul!(g, Xtw, ps)
    g .-= c
    penalty_gradient!(g, reg, b)
end

"""
Let Y be the n x k matrix with i^th row being an indicator of y[i].
That is, Y[i, :] = [0, ..., 1, 0, ..., 0], where Y[i, y[i]] = 1, i = 1:n.

This function conputes: M = transpose(X)*Diagonal(wts)*Y
because it is a constant term in the gradient calculation:

gradient(-LL) = transpose(X)*Diagonal(wts)*(probs .- Y)
              = transpose(X)*Diagonal(wts)*probs .- M
"""
function construct_XtwY(Xtw, y, km1)
    result = fill(0.0, size(Xtw, 1), km1)
    for (i, yi) in enumerate(y)
        yi == 1 && continue
        view(result, :, yi - 1) .+= view(Xtw, :, i)
    end
    result
end

################################################################################
# Hessian

"""
hessian(-LL + penalty)

Let k = the number of categories, and let p = the number of predictors.
The hessian is a (k-1) x (k-1) block matrix, with block size p x p.
In the code below, i and j denote the block indices; i.e., i and j each have k-1 values.
"""
function hessian(X, wts, reg, probs)
    k    = size(probs, 2)
    n, p = size(X)
    H    = fill(0.0, p*(k - 1), p*(k - 1))
    XtW  = fill(0.0, p, n)
    ww   = fill(0.0, n)
    d    = sqrt(eps())
    for j = 1:(k - 1)
        cols = (p*(j - 1) + 1):(p*j)
        for i = j:(k - 1)
            rows = (p*(i - 1) + 1):(p*i)
            hessian_block!(view(H, rows, cols), X, wts, reg, probs, XtW, ww, d, i+1, j+1)
        end
    end
    Hermitian(H, :L)
end

function hessian_block!(H, X, wts, reg, probs, XtW, ww, d, i, j)
    set_working_weights!(ww, wts, probs, d, i, j)
    mul!(XtW, transpose(X), Diagonal(ww))  # W = Diagonal(ww)
    mul!(H, XtW, X)  # H = XtWX
    if i == j
        penalty_hessian!(H, reg)
    end
    nothing
end

"Hessian for j^th group of parameters"
hessian_group!(H, j, X, wts, reg, probs, XtW, ww, d) = hessian_block!(H, X, wts, reg, probs, XtW, ww, d, j+1, j+1)

"w .= wts .* Pi .* (delta_ij - Pj)"
function set_working_weights!(ww, wts, probs, d, i, j)
    Pi = view(probs, :, i)
    if i == j
        ww .= wts .* max.(d, Pi .* (1.0 .- Pi))
    else
        Pj  = view(probs, :, j)
        ww .= wts .* min.(-d, -Pi .* Pj)
    end
end

function set_working_weights!(ww, wts::Nothing, probs, d, i, j)
    Pi = view(probs, :, i)
    if i == j
        ww .= max.(d, Pi .* (1.0 .- Pi))
    else
        Pj  = view(probs, :, j)
        ww .= min.(-d, -Pi .* Pj)
    end
end

end