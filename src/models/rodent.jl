export Rodent, ConstSpecRodent

"""
    Rodent{T}(p::Gaussian{T}, e::CMeanGaussian{T}, d::CMeanGaussian{T})

Variational Auto-Encoder with shared variances.
Provides a constructor that creates a VAE with ARD prior and an ODE decoder.

# Example
With a 2nd order ODE decoder you can solve a harmonic ODE with ξ̇=Wξ+b. 
Setting `W = [0 1; -1 0]`; `b = [0,0]`; `ξ₀=[0,1]` will create a sine wave.
All ODE params are collected in z = {W,b,ξ₀}.

```julia-repl
julia> xlen = 5; tspan = (0f0, Float32(2π)); order = 2; zlen = order^2+order*2;
julia> z = Float32.([0, 1, -1, 0, 0, 0, 0, 1]);
julia> encoder = Dense(xlen, length(z));
julia> decoder = ODEDecoder(order, xlen, tspan);

julia> rodent = Rodent(xlen, zlen, encoder, decoder)
Rodent{Float32}:
 prior   = (Gaussian{Float32}(μ=8-element NoGradArray{Float32,1}, σ2=8-elemen...)
 encoder = (CMeanGaussian{Float32}(mapping=Dense(5, 8), σ2=8-element Array{Flo...)
 decoder = (CMeanGaussian{Float32}(mapping=(ODEDecoder(2, 5, Float32[0.0, 1.570...)

julia> mean(rodent.decoder, z)
5-element Array{Float32,1}:
  0.0
 -1.0000039
  6.377697e-6
  1.000012
 -8.059293e-5
```
"""
struct Rodent{T} <: AbstractVAE{T}
    prior::Gaussian
    encoder::CMeanGaussian
    decoder::CMeanGaussian
end

Flux.@functor Rodent

Rodent(p::Gaussian{T}, e::CMeanGaussian{T}, d::CMeanGaussian{T}) where T =
    Rodent{T}(p, e, d)

function Rodent(xlen::Int, zlen::Int, encoder, decoder, T=Float32)
    λ2z = ones(T, zlen)
    μpz = NoGradArray(zeros(T, zlen))
    prior = Gaussian(μpz, λ2z)

    σ2z = ones(T, zlen)
    enc_dist = CMeanGaussian{T,DiagVar}(encoder, σ2z)

    σ2x = ones(T, 1)
    dec_dist = CMeanGaussian{T,ScalarVar}(decoder, σ2x, xlen)

    Rodent{T}(prior, enc_dist, dec_dist)
end

struct ConstSpecRodent{T} <: AbstractVAE{T}
    const_prior::Gaussian
    spec_prior::Gaussian
    encoder::ConstSpecGaussian
    decoder::CMeanGaussian
end

ConstSpecRodent(cp::Gaussian{T}, sp::Gaussian{T}, e::ConstSpecGaussian{T}, d::CMeanGaussian{T}) where T =
    ConstSpecRodent{T}(cp,sp,e,d)

Flux.@functor ConstSpecRodent

function elbo(m::ConstSpecRodent, x::AbstractArray)
    cz = rand(m.encoder.cnst) 
    sz = rand(m.encoder.spec, x)
    z  = cz .+ sz

    llh = sum(-loglikelihood(m.decoder, x, z))
    ckl = sum(kld(m.encoder.cnst, m.const_prior))
    skl = sum(kld(m.encoder.spec, m.spec_prior, sz))

    llh + ckl + skl
end

function Base.show(io::IO, m::ConstSpecRodent)
    msg = """$(typeof(m)):
     const_prior = $(summary(m.const_prior)))
     spec_prior = $(summary(m.spec_prior))
     encoder = $(summary(m.encoder))
     decoder = $(summary(m.decoder))
    """
    print(io, msg)
end
