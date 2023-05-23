module coordinate_descent

export coordinatedescent

using LinearAlgebra
using Logging

using ..regularization

function coordinatedescent(y, X, wts::Union{Nothing, AbstractVector}, reg::Union{Nothing, ElasticNet}, opts)
    nclasses = length(unique(y))
    nobs     = size(X, 1)
    probs    = fill(0.0, nobs, nclasses)
    opts     = construct_options(opts)
    j        = intercept_index(X)
    loss, B  = _coordinatedescent(y, X, wts, j, probs, reg, opts[:iterations], opts[:f_abstol])
    vcov     = construct_vcov(probs, B, X, wts, reg)
    loss, B, vcov
end

function construct_options(opts)
    default_opts = Dict(:iterations => 250, :f_abstol => 1e-9)
    isnothing(opts) && return default_opts
    isempty(opts)   && return default_opts
    opts2 = Dict{Symbol, valtype(opts)}(Symbol(k) => v for (k, v) in opts)
    merge!(default_opts, opts2)
end

"""
Returns the column index of the intercept if it exists, 0 otherwise.
The intercept column contains only 1s.
"""
function intercept_index(X)
    ni, nj = size(X)
    for j = 1:nj
        vw = view(X, :, j)
        for (i, x) in enumerate(vw)
            x != 1.0 && break
            i == ni && return j
        end
    end
    0
end

function construct_vcov(probs, B, X, wts, reg)
    update_probs!(probs, B, X)
    H = hessian(X, wts, probs)  # hessian(-LL), not hessian(-LL + penalty)
    if rank(H) == length(B)
        if !isnothing(reg)
            @warn "Regularisation implies that the covariance matrix and standard errors are not estimated from MLEs"
        end
        return Matrix(Hermitian(inv(bunchkaufman!(H))))  # varcov(b) = inv(FisherInformation) = inv(Hessian(-LL))
    else
        @warn "Standard errors cannot be computed (Hessian does not have full rank). Check for linearly dependent predictors."
        return fill(NaN, length(B), length(B))
    end
end

################################################################################
# Coordinate Descent

function _coordinatedescent(y, X, wts, intercept_index, probs, reg, iterations, f_tol)
    k    = size(probs, 2)
    n, p = size(X)
    B    = fill(0.0, p, k - 1)
    dB   = fill(0.0, p, k - 1)  # dB = search direction = inv(H)*G. Bnew = B - dB.
    G    = fill(0.0, p, k - 1)  # G = gradient(-LL)/n + penalty_gradient = Xt*(probs .- Y)/n + penalty_gradient
    H    = fill(0.0, p, p)      # H = Hessian for 1 class at a time = Xt * Diagonal(working weights) * X + penalty_hessian
    W    = fill(0.0, n, k - 1)  # W = Working weights = wts .* p .* (1 .- p)
    rs   = filter!((x) -> x != intercept_index, collect(1:p))  # Row indices of B and G that are not the intercept term
    Bvw  = isnothing(reg) ? B : view(B, rs, :)   # The intercept is not penalized
    Gvw  = isnothing(reg) ? G : view(G, rs, :)   # The intercept is not penalized
    Hvw  = isnothing(reg) ? H : view(H, rs, rs)  # The intercept is not penalized
    Xt   = transpose(X)
    XtW  = fill(0.0, p, n)  # Xt * Diagonal(working weights)
    probsview = view(probs, :, 2:k)
    converged = false
    loss_prev = Inf  # Required for the warning message if convergence is not achieved
    update_probs!(probs, B, X)
    loss = -loglikelihood(y, probs, wts)/n + penalty(reg, B)
    d = sqrt(eps())
    for iter = 1:iterations
        # Set search direction: dB = inv(H)*G
        set_working_weights!(W, probsview, wts, d)  # W is modified using probs as input
        probs_minus_y!(probs, y)  # probs -> probs .- Y
        gradient!(G, wts, Xt, XtW, probsview, reg, Bvw, Gvw, n)  # Populate G. Temporarily repurpose XtW = transpose(X)*diagonal(wts).
        for j = 2:k  # Class 1 is the reference class; its coefs are implicitly all 0
            hessian!(H, j, XtW, Xt, W, X, n, reg, Hvw)  # H(-LL/n + penalty) for class j
            update_dB!(dB, j, G, H)
        end

        # Line search along dB
        a = 1.0
        loss_prev = loss
        for i_linesearch = 1:52  # eps() == 1 / (2^52)
            # Update B
            if i_linesearch == 1
                B .-= dB  # The minus is due to the negative gradient used to obtain dB
            else
                B .+= a .* dB  # B = B + a_old*dB - a_new*dB = B + a_new*dB, since a_old = 2*a_new
            end

            # Update loss
            update_probs!(probs, B, X)
            loss = -loglikelihood(y, probs, wts)/n + penalty(reg, B)
            loss < loss_prev && break
            a *= 0.5  # Smaller step size
        end

        # Check for convergence
        converged = isapprox(loss, loss_prev; atol=f_tol) || iszero(loss_prev)
        converged && break
    end
    !converged && @warn "Coordinate Descent did not converge with tolerance $(f_tol). The last change in loss was $(abs(loss - loss_prev))"
    loss, B
end

"Update the column of dB corresponding to class j, where j > 1."
function update_dB!(dB, j, G, H)
    ldiv!(view(dB, :, j - 1), cholesky!(Hermitian(H)), view(G, :, j - 1))  # Or: ldiv!(dB_j, qr!(H), G_j)
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

function set_working_weights!(W, probs, wts, d)
    k = size(probs, 2)
    for j = 1:k
        set_working_weights!(view(W, :, j), probs, wts, d, j, j)
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

"""
gradient(-LL + penalty) = gradient(-LL) + gradient(penalty)

gradient(-LL) = transpose(X)*diagonal(wts)*(probs .- Y)
"""
function gradient!(G, wts, Xt, XtW, probs_minus_Y, reg, Bvw, Gvw, n)
    negativeLL_gradient!(G, XtW, probs_minus_Y, Xt, wts)
    G ./= n  # G(-LL)/n
    penalty_gradient!(Gvw, reg, Bvw)  # The views exclude the intercept parameters (which are not regularized)
end

function negativeLL_gradient!(G, XtW, probs_minus_Y, Xt, wts)
    mul!(XtW, Xt, Diagonal(wts))
    mul!(G, XtW, probs_minus_Y)
end

function negativeLL_gradient!(G, XtW, probs_minus_Y, Xt, wts::Nothing)
    mul!(G, Xt, probs_minus_Y)
end

"""
Hessian for class j parameters.
H(-LL/n + penalty) = H(-LL)/n + H(penalty).
Called during the optimization.
"""
function hessian!(H, j, XtW, Xt, W, X, n, reg, Hvw)
    multiply_3_matrices!(H, XtW, Xt, Diagonal(view(W, :, j - 1)), X)  # Populate H for class j
    H ./= n  # H(-LL)/n
    penalty_hessian!(Hvw, reg) # Adds the hessian of the penalty to H
end

"""
hessian(-LL), not hessian(-LL/n + penalty).

Used to construct vcov.

Let k = the number of categories, and let p = the number of predictors.
The hessian is a (k-1) x (k-1) block matrix, with block size p x p.
In the code below, i and j denote the block indices; i.e., i and j each have k-1 values.
"""
function hessian(X, wts, probs)
    k    = size(probs, 2)
    n, p = size(X)
    H    = fill(0.0, p*(k - 1), p*(k - 1))
    XtW  = fill(0.0, p, n)
    w    = fill(0.0, n)
    Xt   = transpose(X)
    d    = sqrt(eps())
    for j = 1:(k - 1)
        for i = j:(k - 1)
            rows = (p*(i - 1) + 1):(p*i)
            cols = (p*(j - 1) + 1):(p*j)
            set_working_weights!(w, probs, wts, d, i+1, j+1)
            multiply_3_matrices!(view(H, rows, cols), XtW, Xt, Diagonal(w), X)
        end
    end
    Hermitian(H, :L)
end

end