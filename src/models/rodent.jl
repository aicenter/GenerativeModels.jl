export Rodent
export make_ode_decoder

"""
    Rodent{T}(p::Gaussian{T}, e::CMeanGaussian{T}, d::CMeanGaussian{T})

Variational Auto-Encoder with shared variances.
Provides a constructor that creates a VAE with ARD prior and an ODE decoder.

# Example
With a 2nd order ODE decoder you can solve a harmonic ODE with ξ̇=Wξ+b. 
Setting `W = [0 1; -1 0]`; `b = [0,0]`; `ξ₀=[0,1]` will create a sine wave.
All ODE params are collected in z = {W,b,ξ₀}.

```julia-repl
julia> xlen = 5; tspan = (0f0, Float32(2π)); order = 2;

julia> z = Float32.([0, 1, -1, 0, 0, 0, 0, 1]);

julia> encoder = Dense(xlen, length(z))

julia> rodent = Rodent(xlen, encoder, tspan, order)

julia> mean(rodent.decoder, z)
Tracked 5-element Array{Float32,1}:
  0.0f0
  1.0000039f0
 -7.0631504f-6
 -1.0000119f0
  8.059293f-5
```
"""
struct Rodent{T} <: AbstractVAE{T}
    prior::Gaussian
    encoder::CMeanGaussian
    decoder::CMeanGaussian
end

Flux.@functor Rodent

Rodent(p::Gaussian{T}, e::CMeanGaussian{T}, d::CMeanGaussian{T}) where T = Rodent{T}(p, e, d)

function Rodent(xlen::Int, zlen::Int, encoder, decoder, T=Float32)
    λ2z = ones(T, zlen)
    μpz = SVector{zlen}(zeros(T, zlen))
    prior = Gaussian(μpz, λ2z)

    σ2z = ones(T, zlen)
    enc_dist = CMeanGaussian{T}(encoder, σ2z)

    σ2x = ones(T, 1)
    dec_dist = CMeanGaussian{T}(ecoder, σ2x)

    Rodent{T}(prior, enc_dist, dec_dist)
end
