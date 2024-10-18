module regularization

export AbstractRegularizer, penalty, penalty_gradient!, penalty_hessian!, L1, L2, ElasticNet

import LinearAlgebra: norm

abstract type AbstractRegularizer end

penalty(reg, B) = 0.0
penalty_gradient!(gradB, reg, B) = nothing
penalty_hessian!(H, reg) = nothing

################################################################################
# L1

struct L1 <: AbstractRegularizer
    gamma::Float64

    function L1(gamma)
        gamma <= 0.0 && error("Gamma is $(gamma). It should be strictly positive.")
        new(gamma)
    end
end

penalty(reg::L1, B) = reg.gamma * norm(B, 1)

function penalty_gradient!(gradB, reg::L1, B)
    gamma = reg.gamma
    for (i, x) in enumerate(B)
        gradB[i] += x < 0 ? -gamma : gamma
    end
    nothing
end

################################################################################
# L2

struct L2 <: AbstractRegularizer
    lambda::Float64

    function L2(lambda)
        lambda <= 0.0 && error("Lambda is $(lambda). It should be strictly positive.")
        new(lambda)
    end
end

function penalty(reg::L2, B)
    lambda  = reg.lambda
    B_2norm = norm(B, 2)
    0.5 * lambda * B_2norm * B_2norm
end

function penalty_gradient!(gradB, reg::L2, B)
    lambda  = reg.lambda
    gradB .+= lambda .* B
    nothing
end

function penalty_hessian!(H, reg::L2)
    lambda = reg.lambda
    n = size(H, 1)
    for i = 1:n
        @inbounds H[i, i] += lambda
    end
    nothing
end

################################################################################
# ElasticNet

struct ElasticNet <: AbstractRegularizer
    l1::L1
    l2::L2
end

ElasticNet(gamma::Real, lambda::Real) = ElasticNet(L1(gamma), L2(lambda))

penalty(reg::ElasticNet, B) = penalty(reg.l1, B) + penalty(reg.l2, B)

function penalty_gradient!(gradB, reg::ElasticNet, B)
    penalty_gradient!(gradB, reg.l1, B)
    penalty_gradient!(gradB, reg.l2, B)
end

penalty_hessian!(H, reg::ElasticNet) = penalty_hessian!(H, reg.l2)

end