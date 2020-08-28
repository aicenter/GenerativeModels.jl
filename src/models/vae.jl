export VAE
export elbo, mmd_mean, mmd_rand

"""
    VAE{P<:CMD,E<:ACD,D<:ACD}([zlength::Int,p::P] ,e::E ,d::D)

Variational Auto-Encoder.

# Arguments
* `p`: Prior p(z)
* `e`: Encoder p(z|x)
* `d`: Decoder p(x|z)

# Example
Create a vanilla VAE with mappings `f` (R⁵->(R²,R²)) and `g` (R²->(R⁵,R)) that
return mean and variance of appropriate sizes. The output of `f` (and `g`) is
passed to `MvNormal` via the `condition` function of `ConditionalDists`.
```julia-repl
julia> f   = SplitLayer(5, [2,2])
julia> enc = ConditionalMvNormal(f)
julia> g   = SplitLayer(2, [5,1])
julia> dec = ConditionalMvNormal(g)
julia> vae = VAE(2, enc, dec)
VAE:

julia> mean(vae.decoder, mean(vae.encoder, rand(5)))
5×1 Array{Float32,2}:
 -0.26742023
 -0.7905855
 -0.29494995
  0.1694059
  1.123661
```
"""
struct VAE{P<:MvNormal,E<:ConditionalMvNormal,D<:ConditionalMvNormal} <: AbstractVAE
    prior::P
    encoder::E
    decoder::D
end

Flux.@functor VAE

function Flux.trainable(m::VAE)
    (encoder=m.encoder, decoder=m.decoder)
end

function VAE(zlength::Int, enc::ConditionalMvNormal, dec::ConditionalMvNormal)
    W = first(Flux.params(enc))
    μ = fill!(similar(W, zlength), 0)
    σ = fill!(similar(W, zlength), 1)
    prior = MvNormal(μ, σ)
    VAE(prior, enc, dec)
end

"""
    elbo(m::AbstractVAE, x::AbstractArray; β=1)

Evidence lower boundary of the VAE model. `β` scales the KLD term.
"""
function elbo(m::AbstractVAE, x::AbstractArray; β=1)
    # sample latent
    z = rand(m.encoder, x)

    # reconstruction error
    llh = mean(logpdf(m.decoder, x, z))

    # KLD with `condition`ed encoder
    kld = mean(kl_divergence(condition(m.encoder, x), m.prior))

    llh - β*kld
end

# mmd via IPMeasures
# """
#     mmd_mean(m::AbstractVAE, x::AbstractArray, k[; distance])
# 
# Maximum mean discrepancy of a VAE given data `x` and kernel function `k(x,y)`. Uses mean of encoded data.
# """
# mmd_mean(m::AbstractVAE, x::AbstractArray, k; distance = IPMeasures.pairwisel2) = 
#     mmd(k, mean(m.encoder, x), rand(m.prior, size(x, 2)), distance)
# 
# """
#     mmd_rand(m::AbstractVAE, x::AbstractArray, k[; distance])
# 
# Maximum mean discrepancy of a VAE given data `x` and kernel function `k(x,y)`. Samples from the encoder.
# """
# mmd_rand(m::AbstractVAE, x::AbstractArray, k; distance = IPMeasures.pairwisel2) = 
#     mmd(k, rand(m.encoder, x), rand(m.prior, size(x, 2)), distance)

function Base.show(io::IO, m::AbstractVAE)
    p = repr(m.prior)
    p = sizeof(p)>70 ? "($(p[1:70-3])...)" : p
    e = repr(m.encoder)
    e = sizeof(e)>70 ? "($(e[1:70-3])...)" : e
    d = repr(m.decoder)
    d = sizeof(d)>70 ? "($(d[1:70-3])...)" : d
    msg = """$(nameof(typeof(m))):
     prior   = $(p)
     encoder = $(e)
     decoder = $(d)
    """
    print(io, msg)
end


# TODO: use IPMeasures instead
using Distances
function _kld_gaussian(μ1::AbstractArray, σ1::AbstractArray, μ2::AbstractArray, σ2::AbstractArray)
    k  = size(μ1, 1)
    m1 = sum(σ1 ./ σ2, dims=1)
    m2 = sum((μ2 .- μ1).^2 ./ σ2, dims=1)
    m3 = sum(log.(σ2 ./ σ1), dims=1)
    (m1 .+ m2 .+ m3 .- k) ./ 2
end

function (m::Distances.KLDivergence)(p, q)
    μ1, σ1 = mean(p), var(p)
    μ2, σ2 = mean(q), var(q)
    _kld_gaussian(μ1, σ1, μ2, σ2)
end

Distances.kl_divergence(p, q) = KLDivergence()(p, q)
