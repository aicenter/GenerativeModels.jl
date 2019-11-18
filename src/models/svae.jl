export SVAE, SVAE_vmf_prior, SVAE_hsu_prior

"""
    SVAE{T}([prior::Union{HypersphericalUniform{T}, VonMisesFisher{T}}, zlen::Int] encoder::AbstractCVMF, decoder::AbstractCPDF)

HyperSpherical Variational Auto-Encoder.

# Example
Create an S-VAE with either HSU prior or VMF prior with μ = [1, 0, ..., 0] and κ = 1 with:
```julia-repl
julia> enc = CMeanVarVMF{Float32}(Dense(5,4), 3)
CMeanVarVMF{Float32}(mapping=Dense(5, 4), μ_from_hidden=Chain(Dense(4, 3), #51), κ_from_hidden=Dense(4, 1, #52))

julia> dec = CMeanVarGaussian{Float32,ScalarVar}(Dense(3, 6))
CMeanVarGaussian{Float32,ScalarVar}(mapping=Dense(3, 6))

julia> svae = SVAE(HypersphericalUniform{Float32}(3), enc, dec)
SVAE{Float32}:
 prior   = HypersphericalUniform{Float32}(3)
 encoder = (CMeanVarVMF{Float32}(mapping=Dense(5, 4), μ_from_hidden=Chain(Dens...)
 decoder = CMeanVarGaussian{Float32,ScalarVar}(mapping=Dense(3, 6))

julia> mean(svae.decoder, mean(svae.encoder, rand(5, 1)))
5×1 Array{Float32,2}:
 -0.7267006  
  0.6847478  
 -0.032789093
  0.13542232 
 -0.270345421

julia> elbo(svae, rand(Float32, 5, 1))
15.011719478567946
```
"""
struct SVAE{T} <: AbstractSVAE{T}
    prior::Union{HypersphericalUniform{T}, VonMisesFisher{T}}
    encoder::AbstractCVMF{T}
    decoder::AbstractCPDF{T}
end

Flux.@functor SVAE

# SVAE(p::Union{HypersphericalUniform{T}, VonMisesFisher{T}}, e::AbstractCVMF{T}, d::AbstractCPDF{T}) where T = SVAE{T}(p, e, d)

function SVAE_vmf_prior(zlength::Int, enc::AbstractCPDF{T}, dec::AbstractCPDF{T}) where T
    μp = NoGradArray(zeros(T, zlength))
    μp[1] = T(1)
    κp = NoGradArray(ones(T, 1))
    prior = VonMisesFisher(μp, κp)
    SVAE{T}(prior, enc, dec)
end

function SVAE_hsu_prior(zlength::Int, enc::AbstractCPDF{T}, dec::AbstractCPDF{T}) where T
    prior = HypersphericalUniform{T}(zlength)
    SVAE{T}(prior, enc, dec)
end

"""
    elbo(m::SVAE, x::AbstractArray; β=1)

Evidence lower boundary of the SVAE model. `β` scales the KLD term. (Assumes hyperspherical uniform prior)
"""
function elbo(m::SVAE{T}, x::AbstractArray{T}; β=T(1)) where {T}
    z = rand(m.encoder, x)
    llh = mean(-loglikelihood(m.decoder, x, z))
    kl  = mean(kld(m.encoder, m.prior, x))
    llh + β*kl
end

"""
    mmd(m::SVAE, x::AbstractArray, k)

Maximum mean discrepancy of a SVAE model given data `x` and kernel function `k(x,y)`.
"""
mmd(m::SVAE{T}, x::AbstractArray{T}, k) where {T} = mmd(m.encoder, m.prior, x, k) 

function Base.show(io::IO, m::SVAE{T}) where T
    p = short_repr(m.prior, 70)
    e = short_repr(m.encoder, 70)
    d = short_repr(m.decoder, 70)
    msg = """$(typeof(m)):
     prior   = $(p)
     encoder = $(e)
     decoder = $(d)
    """
    print(io, msg)
end