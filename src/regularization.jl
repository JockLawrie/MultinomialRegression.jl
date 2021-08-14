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

function regularize(reg::L1, B, gradB)
    gamma = reg.gamma
    for (i, x) in enumerate(B)
        gradB[i] = x < 0 ? -gamma : gamma
    end
    gamma * sum(abs(x) for x in B)
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

function regularize(reg::L2, B, gradB)
    lambda = reg.lambda
    gradB .= lambda .* B
    0.5 * lambda * sum(x^2 for x in B)
end

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

# Called by the solver
function regularize!(outB, inB, reg::BoxRegularizer, gradB)
    lb    = reg.lowerbound
    width = reg.upperbound - lb
    for (i, x) in enumerate(inB)
        pw       = width / (1.0 + exp(-x))  # p*w, where p = 1/(1+exp(-x))
        outB[i]  = lb + pw
        gradB[i] = pw - pw*pw/width  # w * p * (1 - p)
    end
end

#=
   Called after the solver, for returning values in the specified box.
   Same as previous version but excludes updating the gradient.
=# 
function regularize!(B, reg::BoxRegularizer)
    lb    = reg.lowerbound
    width = reg.upperbound - lb
    for (i, x) in enumerate(B)
        B[i] = lb + width / (1.0 + exp(-x))
    end
end

end