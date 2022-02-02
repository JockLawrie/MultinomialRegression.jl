module regularization

export AbstractRegularizer, L1, L2, penalty

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

function penalty(reg::L1, B, gradB)
    gamma = reg.gamma
    for (i, x) in enumerate(B)
        gradB[i] = x < 0 ? -gamma : gamma
    end
    gamma * norm(B, 1)
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

function penalty(reg::L2, B, gradB)
    lambda = reg.lambda
    gradB .= lambda .* B
    0.5 * lambda * norm(B, 2)^2
end

end