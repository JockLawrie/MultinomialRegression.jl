module regularization

export regularize, AbstractRegularizer, L1, L2

abstract type AbstractRegularizer end

struct L1 <: AbstractRegularizer
    gamma::Float64

    function L1(gamma)
        gamma <= 0.0 && error("Gamma is $(gamma). It should be strictly positive.")
        new(gamma)
    end
end

regularize(reg::L1, B) = reg.gamma * sum(abs(x) for x in B)

struct L2 <: AbstractRegularizer
    lambda::Float64

    function L2(gamma)
        lambda <= 0.0 && error("Lambda is $(lambda). It should be strictly positive.")
        new(lambda)
    end
end

regularize(reg::L2, B) = reg.lambda * sum(x^2 for x in B)


end