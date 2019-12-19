export GAN
export generator_loss, discriminator_loss

"""
    GAN{P<:AbstractPDF,G<:AbstractCPDF,D<:AbstractCPDF}([zlength::Int,p::P], g::G, d::D)

The Generative Adversarial Network.

# Arguments
* `zlength`: Length of latent vector
* `p`: Prior
* `g`: Generator
* `d`: Discriminator

# Example
Create a GAN with standard normal prior with:
```julia-repl
julia> gen = CMeanGaussian{DiagVar}(Dense(2,4),NoGradArray(ones(Float32,4)))
CMeanGaussian{DiagVar}(mapping=Dense(2, 4), σ2=4-element Array{Float32,1}

julia> disc = CMeanGaussian{DiagVar}(Dense(4,1,σ),NoGradArray(ones(Float32,1)))
CMeanGaussian{DiagVar}(mapping=Dense(4, 1, σ), σ2=1-element Array{Float32,1}

julia> gan = GAN(4, gen, disc)
GAN:
 prior   = (Gaussian(μ=4-element Array{Float32,1}, σ2=4-element Array{Float32...)
 generator = (CMeanGaussian{DiagVar}(mapping=Dense(2, 4), σ2=4-element Array{Flo...)
 discriminator = (CMeanGaussian{DiagVar}(mapping=Dense(4, 1, σ), σ2=1-element Array...)
```
"""
struct GAN{P<:AbstractPDF,G<:AbstractCPDF,D<:AbstractCPDF} <: AbstractGAN
	prior::P
	generator::G
	discriminator::D
end

Flux.@functor GAN

GAN(p::P, g::G, d::D) where {P,G,D} = GAN{P,G,D}(p,g,d)

function GAN(zlength::Int, g::AbstractCPDF, d::AbstractCPDF)
    T = eltype(first(params(g)))
    μ = NoGradArray(zeros(T, zlength))
    σ = NoGradArray(ones(T, zlength))
    prior = Gaussian(μ, σ)
    GAN(prior, g, d)
end

"""
	generator_loss(m::GAN, z::AbstractArray)
	generator_loss(m::GAN, batchsize::Int)

Loss of the GAN generator. The input is either the random code `z` or
`batchsize` of samples to generate from the model prior and compute the loss
from.
"""
function generator_loss(m::GAN, z::AbstractArray)
    T = eltype(m.prior.μ)
    generator_loss(T, m.discriminator.mapping(mean(m.generator,z)))
end

generator_loss(m::GAN, batchsize::Int) =
    generator_loss(m, rand(m.prior, batchsize))

"""
	discriminator_loss(m::GAN, x::AbstractArray[, z::AbstractArray])

Loss of the GAN discriminator given a batch of training samples `x` and latent
prior samples `z`.  If z is not given, it is automatically generated from the
model prior.
"""
function discriminator_loss(m::GAN, x::AbstractArray, z::AbstractArray)
    T = eltype(m.prior.μ)
    st = mean(m.discriminator,x)
    sg = mean(m.discriminator, m.generator.mapping(z))
    discriminator_loss(T, st, sg)
end

discriminator_loss(m::GAN, x::AbstractArray) =
    discriminator_loss(m, x, rand(m.prior, size(x,2)))

function Base.show(io::IO, m::AbstractGAN)
    p = repr(m.prior)
    p = sizeof(p)>70 ? "($(p[1:70-3])...)" : p
    g = repr(m.generator)
    g = sizeof(g)>70 ? "($(g[1:70-3])...)" : g
    d = repr(m.discriminator)
    d = sizeof(d)>70 ? "($(d[1:70-3])...)" : d
    msg = """$(nameof(typeof(m))):
     prior   = $(p)
     generator = $(g)
     discriminator = $(d)
    """
    print(io, msg)
end
