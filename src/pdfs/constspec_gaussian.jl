export loglikelihood, kld, rand
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

spec_mean(p::ConstSpecGaussian, z::AbstractArray) = mean(p.spec, z)
spec_variance(p::ConstSpecGaussian, z::AbstractArray) = variance(p.spec, z)
spec_mean_var(p::ConstSpecGaussian, z::AbstractArray) = mean_var(p.spec, z)

mean(p::ConstSpecGaussian, z::AbstractArray) = vcat(repeat(const_mean(p), batch), spec_mean(p,z))
variance(p::ConstSpecGaussian, z::AbstractArray) = vcat(...)
mean_var(p::ConstSpecGaussian, z::AbstractArray) = vcat(...)

function rand(p::ConstSpecGaussian, z::AbstractArray)
    cr = rand(p.cnst)
    sr = rand(p.spec, z)
    vcat(repeat(cr, batch), sr)
end

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
