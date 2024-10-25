module optim

export fit_optim

using LinearAlgebra
using Logging

using ..blockwise_coord_descent
using ..regularization

function fit_optim(y, X, wts::Union{Nothing, AbstractVector}=nothing,
                   reg::Union{Nothing, AbstractRegularizer}=nothing,
                   opts::Union{Nothing, T}=nothing) where{T}
    nclasses = length(unique(y))
    opts     = override_default_options(defaultopts(), opts)  # User-specified options override default options
    Xs       = [view(X, :, :) for k = 1:(nclasses - 1)]
    cache    = init_cache(y, Xs, wts, reg)
    loss, B  = blockwise_coordinate_descent(loss!, block_gradient!, block_hessian!, y, Xs, wts, opts, cache)
    cache    = (reg=nothing, probs=cache.probs, ww=cache.ww, XtW=cache.XtW)
    update_probs!(cache.probs, B, X)
    H = hessian(X, wts, cache)  # hessian(-LL), not hessian(-LL + penalty)
    B = reshape(vcat(B...), size(X, 2), nclasses-1)  # Convert Vector{Vector} to Matrix
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
# Cache

function init_cache(y, Xs, w, reg)
    # Cache for calculating loss
    n        = size(Xs[1], 1)
    nclasses = length(unique(y))
    probs    = fill(0.0, n, nclasses)
    # Cache for calculating the gradient
    wp_minus_y = fill(0.0, n)  # Used for gradient
    # Cache for calculating the hessian
    ww   = fill(0.0, n)  # Working weights. H = Xt*Diagonal(ww)*X
    pmax = maximum(size(X, 2) for X in Xs)
    XtW  = fill(0.0, pmax, n)
    (reg=reg, probs=probs, wp_minus_y=wp_minus_y, ww=ww, XtW=XtW)
end

################################################################################
# Loss function

"Modifies: θ, cache.probs"
function loss!(y, Xs, w, θ, cache)
    update_probs!(cache.probs, θ, Xs[1])
    -loglikelihood(y, cache.probs, w) + penalty(cache.reg, θ)
end

loglikelihood(y, probs, w)          = @inbounds sum(w[i]*log(max(probs[i, yi], 1e-12)) for (i, yi) in enumerate(y))
loglikelihood(y, probs, w::Nothing) = @inbounds sum(     log(max(probs[i, yi], 1e-12)) for (i, yi) in enumerate(y))

function update_probs!(probs, θ, X)
    fill!(view(probs, :, 1), 0.0)  # probs[:, 1] = eta[:, 1]
    for (b, θb) in enumerate(θ)
        mul!(view(probs, :, b+1), X, θb)  # probs[:, b+1] = eta[:, b+1], b = 1:nblocks
    end
    rowwise_softmax!(probs)
end

"Transform eta[i, :] to probs[i, :] for i = 1:size(probs, 1)"
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

function block_gradient!(g, b, y, Xs, w, θ, cache)
    populate_wp_minus_y!(cache, y, w, b)
    mul!(g[b], transpose(Xs[b]), cache.wp_minus_y)  # g = transpose(X)*Diagonal(w)*(probs .- Y)
    penalty_gradient!(g[b], cache.reg, θ[b])
end

function populate_wp_minus_y!(cache, y, w, b)
    wp_minus_y = cache.wp_minus_y
    k     = b + 1
    probs = view(cache.probs, :, k)
    if isnothing(w)
        for (i, yi) in enumerate(y)
            wp_minus_y[i] = yi == k ? (probs[i] - 1.0) : probs[i]
        end
    else
        for (i, yi) in enumerate(y)
            wp_minus_y[i] = yi == k ? w[i]*(probs[i] - 1.0) : w[i]*probs[i]
        end
    end
    nothing
end

################################################################################
# Hessian

"Hessian for b^th block of parameters"
block_hessian!(H, b, Xs, w, cache) = hessian_block_ij!(H[b], Xs[b], w, cache, b+1, b+1)

function hessian_block_ij!(H, X, w, cache, i, j)
    set_working_weights!(cache.ww, w, cache.probs, i, j)
    n, p = size(X)
    XtW  = view(cache.XtW, 1:p, 1:n)
    mul!(XtW, transpose(X), Diagonal(cache.ww))  # W = Diagonal(ww)
    mul!(H, XtW, X)  # H = XtWX
    if i == j
        penalty_hessian!(H, cache.reg)
    end
    nothing
end

"ww .= w .* Pi .* (delta_ij - Pj)"
function set_working_weights!(ww, w, probs, i, j)
    d  = sqrt(eps())
    Pi = view(probs, :, i)
    if i == j
        ww .= w .* max.(d, Pi .* (1.0 .- Pi))
    else
        Pj  = view(probs, :, j)
        ww .= w .* min.(-d, -Pi .* Pj)
    end
end

function set_working_weights!(ww, w::Nothing, probs, i, j)
    d  = sqrt(eps())
    Pi = view(probs, :, i)
    if i == j
        ww .= max.(d, Pi .* (1.0 .- Pi))
    else
        Pj  = view(probs, :, j)
        ww .= min.(-d, -Pi .* Pj)
    end
end

"""
hessian(-LL + penalty)

Let k = the number of categories, and let p = the number of predictors.
The hessian is a (k-1) x (k-1) block matrix, with block size p x p.
In the code below, i and j denote the block indices; i.e., i and j each have k-1 values.
"""
function hessian(X, w, cache)
    k    = size(cache.probs, 2)
    n, p = size(X)
    H    = fill(0.0, p*(k - 1), p*(k - 1))
    for j = 1:(k - 1)
        cols = (p*(j - 1) + 1):(p*j)
        for i = j:(k - 1)
            rows = (p*(i - 1) + 1):(p*i)
            hessian_block_ij!(view(H, rows, cols), X, w, cache, i+1, j+1)
        end
    end
    Hermitian(H, :L)
end

end