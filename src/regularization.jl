module regularization

export regularize, regularize!, AbstractRegularizer, L1, L2, BoxRegularizer

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

regularize(reg::L1, B) = reg.gamma * sum(abs(x) for x in B)

################################################################################
# L2

struct L2 <: AbstractRegularizer
    lambda::Float64

    function L2(lambda)
        lambda <= 0.0 && error("Lambda is $(lambda). It should be strictly positive.")
        new(lambda)
    end
end

regularize(reg::L2, B) = 0.5 * reg.lambda * sum(x^2 for x in B)

################################################################################
# BoxRegularizer

"Constrains each parameter in B to be in [lowerbound, upperbound]."
struct BoxRegularizer <: AbstractRegularizer
    lowerbound::Float64
    upperbound::Float64

    function BoxRegularizer(lb, ub)
        lb >= ub && error("The lower bound must be strictly lower than the upper bound.")
        new(lb, ub)
    end
end

function regularize!(outB, inB, reg::BoxRegularizer)
    lb    = reg.lowerbound
    width = reg.upperbound - lb
    for (i, x) in enumerate(inB)
        outB[i] = lb + width / (1.0 + exp(-x))
    end
    0.0
end

end