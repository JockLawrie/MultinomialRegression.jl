module blockwise_coord_descent

export blockwise_coordinate_descent

using LinearAlgebra

"Cyclic blockwise coordinate descent"
function blockwise_coordinate_descent(f, block_gradient!, block_hessian!, y, Xs, w, opts, cache)
    checkinputs(y, Xs, w)
    θ = initθ(Xs)
    s = [fill(0.0, length(block)) for block in θ]  # Search direction: θnew = θ + a*s for some scalar a
    g = [fill(0.0, length(block)) for block in θ]  # Gradient for each block
    H = [fill(0.0, length(block), length(block)) for block in θ]  # Hessian for each block
    loss = f(y, Xs, w, θ, cache)
    iterations = opts[:iterations]
    g_abstol   = opts[:g_abstol]
    converged  = false
    for iter = 1:iterations
        for (b, sb) in enumerate(s)
            block_gradient!(g, b, y, Xs, w, θ, cache)
            block_hessian!(H, b, Xs, w, cache)
            block_searchdirection!(sb, H[b], g[b])
            loss = linesearch!(θ[b], sb, f, y, Xs, w, θ, cache, loss)
        end
        converged = isapprox(maxabs(g), 0.0; atol=g_abstol)
        converged && break
    end
    !converged && @warn "Blockwise Coordinate Descent did not converge: gnorm = $(maxabs(g)) > g_abstol ($(g_abstol))."
    loss, θ
end

"Check that the number of observations in y, Xs and w are the same."
function checkinputs(y, Xs, w)
    n = length(y)
    !isnothing(w) && length(w) != n && error("y has $(n) observations, but w has length $(length(w))")
    for (k, X) in enumerate(Xs)
        size(X, 1) != n && error("y has $(n) observations, but Xs[$(k)] has $(size(X, 1)) rows")
    end
    n
end

function initθ(Xs)
    θ = Vector{typeof(0.0)}[]
    for X in Xs
        p = size(X, 2)
        push!(θ, fill(0.0, p))
    end
    θ
end

"""
For block b, set the search direction s[b] using the Newton-Raphson method.
That is, set s[b] = inv(H[b])*g[b].
Could replace qr!(H) with cholesky!(Hermitian(H)); the former is more stable but slower.
"""
block_searchdirection!(s, H, g) = ldiv!(s, qr!(H), g)

"""
Basic line search, θ -> θ + a*s, for some a.

1. Start with a large step size, a = 2, and a step size multiplier m in (0, 1).
2. Set θb = θb - a*sb
3. If loss(θ, ...) < prevloss, return loss, else a *= m and go back to Step 2.
"""
function linesearch!(θb, sb, f, y, Xs, w, θ, cache, prevloss)
    loss  = Inf
    aprev = 0.0  # Previous step size
    a     = 2.0  # Initial step size
    m     = 0.8  # At each iteration, multiply the step size by m
    while a > 1e-7
        # Update θb
        c = aprev - a
        θb .+= c .* sb  # θb = θb + aprev*sb - a*sb = θb + (aprev - a)*sb = θb + c*sb

        # Update loss
        loss = f(y, Xs, w, θ, cache)
        loss < prevloss && break
        aprev = a
        a *= m  # Smaller step size
    end
    loss
end

"The infinity norm of the gradient"
function maxabs(g)
    result = -Inf
    for gb in g
        v = norm(gb, Inf)
        if v > result
            result = v
        end
    end
    result
end

end