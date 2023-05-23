module regularization

export penalty, penalty_gradient!, penalty_hessian!, ElasticNet

import LinearAlgebra: norm

################################################################################
# No regularization

penalty(reg, B) = 0.0
penalty_gradient!(gradB, reg, B) = nothing
penalty_hessian!(H, reg) = nothing

################################################################################
# Regularization

const FLOAT = typeof(0.0)

"penalty = λ * (0.5*(1-α)*norm(B,2)^2 + α*norm(B,1))"
struct ElasticNet
    lambda::FLOAT
    alpha::FLOAT

    function ElasticNet(lambda, alpha)
        lambda < 0.0  && error("Lambda is $(lambda). It should be strictly positive.")
        lambda == 0.0 && error("Lambda is 0. Set reg=nothing.")
        (alpha < 0.0 || alpha > 1.0) && error("Alpha must be in the interval [0, 1]")
        new(lambda, alpha)
    end
end

function penalty(reg::ElasticNet, B)
    B_2norm = norm(B, 2)
    reg.lambda * (0.5*(1.0-reg.alpha)*B_2norm*B_2norm + reg.alpha*norm(B, 1))
end

"The derivative with respect to element b is λ((1-α)b + αsign(b))"
function penalty_gradient!(gradB, reg::ElasticNet, B)
    lambda = reg.lambda
    alpha  = reg.alpha
    one_minus_alpha = 1.0 - alpha
    for (i, b) in enumerate(B)
        b == 0.0 && continue
        grad_L1   = b < 0 ? -alpha : alpha
        gradB[i] += lambda*(one_minus_alpha*b + grad_L1)
    end
    nothing
end

"The derivative with respect to b_i and b_j is λ(1-α) if i == j."
function penalty_hessian!(H, reg::ElasticNet)
    m = reg.lambda * (1.0 - reg.alpha)
    n = size(H, 1)
    for i = 1:n
        @inbounds H[i, i] += m
    end
    nothing
end

end