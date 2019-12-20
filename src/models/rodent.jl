export Rodent, ConstSpecRodent

"""
    Rodent{P<:Gaussian,E<:CMeanGaussian,D<:CMeanGaussian}(p::P, e::E, d::D)

Variational Auto-Encoder with shared variances.
Mainly used for the additional construtor that it provides, which creates a VAE
with ARD prior and an ODE decoder.

# Arguments
* `p`: Prior p(z)
* `e`: Encoder p(z|x)
* `d`: Decoder p(x|z)
"""
struct Rodent{P<:Gaussian,E<:CMeanGaussian,D<:CMeanGaussian} <: AbstractVAE
    prior::Gaussian
    encoder::CMeanGaussian
    decoder::CMeanGaussian
end

Flux.@functor Rodent

Rodent(p::P, e::E, d::D) where {P,E,D} = Rodent{P,E,D}(p,e,d)

"""
    Rodent(slen::Int, tlen::Int, dt::T, encoder) where T

Constructs a VAE with ARD prior on the latent dimension z and an ODE solver
as decoder. Uses `CMeanGaussian`s for encoder and decoder.

# Arguments
* `slen`: state length of the ODE
* `tlen`: number of timesteps that are returned by decoder
* `dt`: sampling timestep of ODE decoder
* `encoder`: mapping from input `x` to latent code `z`

# Example
With a 2nd order ODE decoder you can describe a harmonic oscillator with dξ=Wξ+b. 
Setting `W = [0 1; -1 0]`; `b = [0,0]`; `ξ₀=[0,1]` will create a sine wave.
All ODE params are collected in z = [W,b,ξ₀]:

```julia-repl
julia> slen = 2;  # state length (order of ODE)
julia> tlen = 5;  # number of timesteps to output from decoder
julia> dt = 2π/tlen;  # timestep
julia> enc = Dense(slen,slen)  # ODE model
julia> z = Float32.([0, 1, -1, 0, 0, 0, 0, 1]);  # latent code to produce clean sine

julia> rodent = Rodent(slen, tlen, dt, enc)
Rodent:
 prior   = (Gaussian{Float32}(μ=8-element NoGradArray{Float32,1}, σ2=8-elemen...)
 encoder = (CMeanGaussian{Float32}(mapping=Dense(5, 8), σ2=8-element Array{Flo...)
 decoder = (CMeanGaussian{Float32}(mapping=(ODEDecoder(2, 5, Float32[0.0, 1.570...)

julia> mean(rodent.decoder, z)
2×5 Array{Float32,2}:
 0.0  -0.999999     8.41825e-6  1.00005     -6.67544e-5
 1.0  -6.49383e-7  -1.0         4.01425e-5   1.00002
```
"""
function Rodent(slen::Int, tlen::Int, dt::T, encoder, observe::Function) where T
    ode = Dense(slen,slen)
    zlen = length(destructure(ode)) + slen
    tspan = (T(0.0), tlen*dt)

    μpz = NoGradArray(zeros(T, zlen))
    λ2z = ones(T, zlen)
    prior = Gaussian(μpz, λ2z)

    σ2z = ones(T, zlen) / 100
    enc_dist = CMeanGaussian{DiagVar}(encoder, σ2z)

    σ2x = ones(T, 1)
    decoder = FluxODEDecoder(slen, tlen, dt, ode, observe)
    dec_dist = CMeanGaussian{ScalarVar}(decoder, σ2x, tlen)

    Rodent(prior, enc_dist, dec_dist)
end

struct ConstSpecRodent{CP<:Gaussian,SP<:Gaussian,E<:ConstSpecGaussian,D<:CMeanGaussian} <: AbstractVAE
    const_prior::CP
    spec_prior::SP
    encoder::E
    decoder::D
end

ConstSpecRodent(cp::CP, sp::SP, e::E, d::D) where {CP,SP,E,D} =
    ConstSpecRodent{CP,SP,E,D}(cp,sp,e,d)

Flux.@functor ConstSpecRodent

function elbo(m::ConstSpecRodent, x::AbstractArray)
    cz = rand(m.encoder.cnst) 
    sz = rand(m.encoder.spec, x)
    z  = cz .+ sz

    llh = sum(-loglikelihood(m.decoder, x, z))
    ckl = sum(kl_divergence(m.encoder.cnst, m.const_prior))
    skl = sum(kl_divergence(m.encoder.spec, m.spec_prior, sz))

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
