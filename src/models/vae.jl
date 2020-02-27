export VAE
export elbo, mmd_mean, mmd_rand

"""
    VAE{P<:CDM,E<:ACD,D<:ACD}([zlength::Int,p::P] ,e::E ,d::D)

Variational Auto-Encoder.

# Arguments
* `p`: Prior p(z)
* `zlength`: Length of latent vector
* `e`: Encoder p(z|x)
* `d`: Decoder p(x|z)

# Example
Create a vanilla VAE with standard normal prior with:
```julia-repl
julia> enc = CMeanVarGaussian{DiagVar}(Dense(5,4))
CMeanVarGaussian{DiagVar}(mapping=Dense(5, 4))

julia> dec = CMeanVarGaussian{ScalarVar}(Dense(2,6))
CMeanVarGaussian{ScalarVar}(mapping=Dense(2, 6))

julia> vae = VAE(2, enc, dec)
VAE:
 prior   = (Gaussian(μ=2-element Array{Float32,1}, σ2=2-element Array{Float32...)
 encoder = CMeanVarGaussian{DiagVar}(mapping=Dense(5, 4))
 decoder = CMeanVarGaussian{ScalarVar}(mapping=Dense(2, 6))

julia> mean(vae.decoder, mean(vae.encoder, rand(5)))
5×1 Array{Float32,2}:
 -0.26742023
 -0.7905855
 -0.29494995
  0.1694059
  1.123661
```
"""
struct VAE{P<:CDM,E<:ACD,D<:ACD} <: AbstractVAE
    prior::P
    encoder::E
    decoder::D
end

Flux.@functor VAE

VAE(p::P, e::E, d::D) where {P,E,D} = VAE{P,E,D}(p,e,d)

function VAE(zlength::Int, enc::ACD, dec::ACD)
    T = eltype(first(params(enc)))
    μp = NoGradArray(zeros(T, zlength))
    σ2p = NoGradArray(ones(T, zlength))
    prior = Gaussian(μp, σ2p)
    VAE(prior, enc, dec)
end

"""
    elbo(m::AbstractVAE, x::AbstractArray; β=1)

Evidence lower boundary of the VAE model. `β` scales the KLD term.
"""
function elbo(m::AbstractVAE, x::AbstractArray; β=1)
    z = rand(m.encoder, x)
    llh = mean(logpdf(m.decoder, x, z))
    kld = mean(kl_divergence(m.encoder, m.prior, x))
    llh - β*kld
end

# mmd via IPMeasures
"""
    mmd_mean(m::AbstractVAE, x::AbstractArray, k[; distance])

Maximum mean discrepancy of a VAE given data `x` and kernel function `k(x,y)`. Uses mean of encoded data.
"""
mmd_mean(m::AbstractVAE, x::AbstractArray, k; distance = IPMeasures.pairwisel2) = 
    mmd(k, mean(m.encoder, x), rand(m.prior, size(x, 2)), distance)

"""
    mmd_rand(m::AbstractVAE, x::AbstractArray, k[; distance])

Maximum mean discrepancy of a VAE given data `x` and kernel function `k(x,y)`. Samples from the encoder.
"""
mmd_rand(m::AbstractVAE, x::AbstractArray, k; distance = IPMeasures.pairwisel2) = 
    mmd(k, rand(m.encoder, x), rand(m.prior, size(x, 2)), distance)

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
