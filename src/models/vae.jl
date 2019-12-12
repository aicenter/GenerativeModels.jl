export VAE
export elbo, mmd_mean, mmd_rand

"""
    VAE{T}([prior::Gaussian, zlen::Int] encoder::AbstractCPDF, decoder::AbstractCPDF)

Variational Auto-Encoder.

# Example
Create a vanilla VAE with standard normal prior with:
```julia-repl
julia> enc = CMeanVarGaussian{Float32,DiagVar}(Dense(5,4))
CMeanVarGaussian{Float32,DiagVar}(mapping=Dense(5, 4))

julia> dec = CMeanVarGaussian{Float32,ScalarVar}(Dense(2,6))
CMeanVarGaussian{Float32,ScalarVar}(mapping=Dense(2, 6))

julia> vae = VAE(2, enc, dec)
VAE{Float32}:
 prior   = (Gaussian{Float32}(μ=2-element NoGradArray{Float32,1}, σ2=2-elemen...)
 encoder = CMeanVarGaussian{Float32,DiagVar}(mapping=Dense(5, 4))
 decoder = CMeanVarGaussian{Float32,ScalarVar}(mapping=Dense(2, 6))

julia> mean(vae.decoder, mean(vae.encoder, rand(5)))
5×1 Array{Float32,2}:
 -0.26742023
 -0.7905855
 -0.29494995
  0.1694059
  1.123661
```
"""
struct VAE{T} <: AbstractVAE{T}
    prior::Gaussian
    encoder::AbstractCPDF
    decoder::AbstractCPDF
end

Flux.@functor VAE

VAE(p::Gaussian{T}, e::AbstractCPDF{T}, d::AbstractCPDF{T}) where T = VAE{T}(p, e, d)

function VAE(zlength::Int, enc::AbstractCPDF{T}, dec::AbstractCPDF{T}) where T
    μp = NoGradArray(zeros(T, zlength))
    σ2p = NoGradArray(ones(T, zlength))
    prior = Gaussian(μp, σ2p)
    VAE{T}(prior, enc, dec)
end

"""
    elbo(m::AbstractVAE, x::AbstractArray; β=1)

Evidence lower boundary of the VAE model. `β` scales the KLD term.
"""
function elbo(m::AbstractVAE, x::AbstractArray; β=1)
    z = rand(m.encoder, x)
    llh = mean(-loglikelihood(m.decoder, x, z))
    kld = mean(kl_divergence(m.encoder, m.prior, x))
    llh + β*kld
end

# mmd via IPMeasures
"""
    mmd_mean(m::AbstractVAE, x::AbstractArray, k[; distance])

Maximum mean discrepancy of a VAE given data `x` and kernel function `k(x,y)`. Uses mean of encoded data.
"""
mmd_mean(m::AbstractVAE, x::AbstractArray, k; distance = IPMeasures.pairwisel2) = 
    mmd(k, mean(m.encoder, x), rand(m.prior, size(x, 2)))

"""
    mmd_rand(m::AbstractVAE, x::AbstractArray, k[; distance])

Maximum mean discrepancy of a VAE given data `x` and kernel function `k(x,y)`. Samples from the encoder.
"""
mmd_rand(m::AbstractVAE, x::AbstractArray, k; distance = IPMeasures.pairwisel2) = 
    mmd(k, rand(m.encoder, x), rand(m.prior, size(x, 2)))

function Base.show(io::IO, m::AbstractVAE{T}) where T
    p = repr(m.prior)
    p = sizeof(p)>70 ? "($(p[1:70-3])...)" : p
    e = repr(m.encoder)
    e = sizeof(e)>70 ? "($(e[1:70-3])...)" : e
    d = repr(m.decoder)
    d = sizeof(d)>70 ? "($(d[1:70-3])...)" : d
    msg = """$(typeof(m)):
     prior   = $(p)
     encoder = $(e)
     decoder = $(d)
    """
    print(io, msg)
end
