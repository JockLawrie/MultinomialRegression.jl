module regularization

export AbstractRegularizer, L1, L2, ElasticNet, penalty, penalty_gradient!

import LinearAlgebra: norm

abstract type AbstractRegularizer end

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

end