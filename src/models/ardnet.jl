export ARDNet

const ACG = ConditionalDists.AbstractConditionalGaussian

const FDCMeanGaussian = CMeanGaussian{V,<:FluxDecoder} where V

"""
    ARDNet(h::InverseGamma, p::Gaussian, e::Gaussian, d::ACGaussian)

Generative model that emposes the sparsifying ARD (*Automatic Relevance
Determination*) prior on the weights of the decoder mapping:

p(x|z) = N(x|ϕ(z),σx²)
p(z)   = N(z|0,diag(λz²))
p(λz)  = iG(λ|α0,β0)

where the posterior on z is a multivariate Gaussian
q(z|x) = N(z|μz,σz²)
"""
struct ARDNet{H<:InverseGamma,P<:Gaussian,E<:Gaussian,D<:ACG} <: AbstractGM
    hyperprior::H
    prior::P
    encoder::E
    decoder::D
end

Flux.@functor ARDNet

function ConditionalDists.logpdf(p::ACG, x::AbstractArray{T}, z::AbstractArray{T},
                                 ps::AbstractVector{T}) where T
    μ = mean(p, z, ps)
    σ2 = var(p, z)
    d = x - μ
    y = d .* d
    y = (1 ./ σ2) .* y .+ log.(σ2) .+ T(log(2π))
    -sum(y, dims=1) / 2
end

ConditionalDists.mean(p::FDCMeanGaussian, z::AbstractArray, ps::AbstractVector) =
    p.mapping(z, ps)

function elbo(m::ARDNet, x, y; β=1)
    ps = reshape(rand(m.encoder),:)
    llh = sum(logpdf(m.decoder, y, x, ps))
    kld = sum(kl_divergence(m.encoder, m.prior))
    lpλ = sum(logpdf(m.hyperprior, var(m.prior)))
    llh - β*(kld - lpλ)
end
