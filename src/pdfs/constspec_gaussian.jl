export loglikelihood
export const_mean, const_variance, const_mean_var
export spec_mean, spec_variance, spec_mean_var
export ConstSpecGaussian

# TODO: maybe rename const parts, as it is not really constant during training?

struct ConstSpecGaussian{T} <: AbstractCPDF{T}
    cnst::AbstractPDF
    spec::AbstractCPDF
end

ConstSpecGaussian(c::AbstractPDF{T}, s::AbstractCPDF{T}) where T = ConstSpecGaussian{T}(c,s)

Flux.@functor ConstSpecGaussian

const_mean(p::ConstSpecGaussian) = mean(p.cnst)
const_variance(p::ConstSpecGaussian) = variance(p.cnst)
const_mean_var(p::ConstSpecGaussian) = mean_var(p.cnst)
const_rand(p::ConstSpecGaussian) = rand(p.cnst)

spec_mean(p::ConstSpecGaussian, z::AbstractArray) = mean(p.spec, z)
spec_variance(p::ConstSpecGaussian, z::AbstractArray) = variance(p.spec, z)
spec_mean_var(p::ConstSpecGaussian, z::AbstractArray) = mean_var(p.spec, z)
spec_rand(p::ConstSpecGaussian, z::AbstractArray) = rand(p.spec, z)

function mean(p::ConstSpecGaussian, z::AbstractArray)
    μc = repeat(const_mean(p), 1, size(z,2))
    μs = spec_mean(p, z)
    (μc, μs)
end

function variance(p::ConstSpecGaussian, z::AbstractArray)
    σc = repeat(const_variance(p), 1, size(z,2))
    σs = spec_variance(p, z)
    (σc, σs)
end

mean_var(p::ConstSpecGaussian, z::AbstractArray) = (mean(p,z), variance(p,z))

rand(p::ConstSpecGaussian, z::AbstractArray) = (const_rand(p), spec_rand(p,z))

function loglikelihood(p::ConstSpecGaussian, x::AbstractArray, z::AbstractArray)
    cllh = loglikelihood(p.cnst, x)
    sllh = loglikelihood(p.spec, x, z)
    cllh + sllh
end

function Base.show(io::IO, p::ConstSpecGaussian)
    msg = """$(typeof(p)):
     const = $(p.cnst)
     spec  = $(p.spec)
    """
    print(io, msg)
end
