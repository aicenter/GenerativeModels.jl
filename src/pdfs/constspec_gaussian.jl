export loglikelihood, kld, rand
export ConstSpecGaussian

struct ConstSpecGaussian
    cnst::AbstractPDF
    spec::AbstractCPDF
end

cnst_mean(p::ConstSpecGaussian) = mean(p.cnst)
cnst_variance(p::ConstSpecGaussian) = variance(p.cnst)
cnst_mean_var(p::ConstSpecGaussian) = mean_var(p.cnst)

spec_mean(p::ConstSpecGaussian, z::AbstractArray) = mean(p.spec, z)
spec_variance(p::ConstSpecGaussian, z::AbstractArray) = variance(p.spec, z)
spec_mean_var(p::ConstSpecGaussian, z::AbstractArray) = mean_var(p.spec, z)

mean(p::ConstSpecGaussian, z::AbstractArray) = cnst_mean(p) .+ spec_mean(p,z)
variance(p::ConstSpecGaussian, z::AbstractArray) = error("Not defined.")
mean_var(p::ConstSpecGaussian, z::AbstractArray) = error("Not defined.")

function rand(p::ConstSpecGaussian, z::AbstractArray)
    cr = rand(p.cnst)
    sr = rand(p.spec, z)
    sr .+ cr
end

function loglikelihood(p::ConstSpecGaussian, x::AbstractArray, z::AbstractArray)
    cllh = loglikelihood(p.cnst, x)
    sllh = loglikelihood(p.spec, x, z)
    cllh + sllh
end

function kld(p::ConstSpecGaussian, q::Gaussian, z::AbstractArray)
    ckl = kld(p.cnst, q)
    skl = kld(p.spec, q, z)
    ckl + skl
end

Flux.@functor ConstSpecGaussian
