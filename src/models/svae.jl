export SVAE

"""
    SVAE{T}([prior::Gaussian, zlen::Int] encoder::AbstractCVMF, decoder::AbstractCPDF)

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
struct SVAE{T} <: AbstractSVAE{T}
    prior # add Union of VMF and nothing as that makes sense and makes it nicer
    encoder::AbstractCVMF
    decoder::AbstractCPDF
end

Flux.@functor SVAE

SVAE(p, e::AbstractCVMF{T}, d::AbstractCPDF{T}) where T = VAE{T}(p, e, d)

function SVAE(zlength::Int, enc::AbstractCPDF{T}, dec::AbstractCPDF{T}) where T
    μp = NoGradArray(zeros(T, zlength))
    σ2p = NoGradArray(ones(T, zlength))
    prior = Gaussian(μp, σ2p)
    VAE{T}(prior, enc, dec)
end

"""
    elbo(m::SVAE, x::AbstractArray; β=1)

Evidence lower boundary of the SVAE model. `β` scales the KLD term. (Assumes hyperspherical uniform prior)
"""
function elbo(m::SVAE, x::AbstractArray; β=1)
    z = rand(m.encoder, x)
    llh = mean(-loglikelihood(m.decoder, x, z))
    kl  = mean(kld(m.encoder, x))
    llh + β*kl
end

"""
    mmd(m::AbstractVAE, x::AbstractArray, k)

Maximum mean discrepancy of a VAE model given data `x` and kernel function `k(x,y)`.
"""
mmd(m::AbstractVAE, x::AbstractArray, k) = error("Not implemented!")

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