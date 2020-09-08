"""
    VAE([zlength::Int,p::P] ,e::E ,d::D)

Variational Auto-Encoder.

# Arguments
* `p`: MvNormal prior p(z)
* `zlength`: length of prior MvNormal distribution
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
struct VAE{P<:ContinuousMultivariateDistribution,E<:ConditionalMvNormal,D<:ConditionalMvNormal} <: AbstractVAE
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
    prior = DistributionsAD.TuringMvNormal(μ, σ)
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

"""
    mmd_mean(m::AbstractVAE, x::AbstractArray, k::Kernel)

Maximum mean discrepancy of a VAE given data `x` and kernel `k`.
Uses mean of encoded data.
"""
function mmd_mean(m::AbstractVAE, x::AbstractArray, k::AbstractKernel = GaussianKernel())
    mmd(k, mean(m.encoder,x), rand(m.prior,size(x,2)))
end

"""
    mmd_rand(m::AbstractVAE, x::AbstractArray, k::Kernel)

Maximum mean discrepancy of a VAE given data `x` and kernel `k`.
Samples from the encoder.
"""
function mmd_rand(m::AbstractVAE, x::AbstractArray, k::AbstractKernel = GaussianKernel())
    mmd(k, rand(m.encoder,x), rand(m.prior,size(x,2)))
end

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
