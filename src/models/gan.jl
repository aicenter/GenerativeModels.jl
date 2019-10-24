export GAN
export generator_loss, discriminator_loss

"""
	GAN{T}([prior::AbstractPDF,] generator::AbstractCPDF, discriminator::AbstractCPDF)

The Generative Adversarial Network.

# Example
Create a GAN with standard normal prior with:
```julia-repl
julia> gen = CMeanGaussian{Float32,DiagVar}(Dense(2,4),NoGradArray(ones(Float32,4)))
CMeanGaussian{Float32}(mapping=Dense(2, 4), σ2=4-element Array{Float32,1}

julia> disc = CMeanGaussian{Float32,DiagVar}(Dense(4,1,σ),NoGradArray(ones(Float32,1)))
CMeanGaussian{Float32}(mapping=Dense(4, 1, σ), σ2=1-element Array{Float32,1}

julia> gan = GAN(4, gen, disc)
 prior   = (Gaussian{Float32}(μ=4-element NoGradArray{Float32,1}, σ2=4-elemen...)
 generator = (CMeanGaussian{Float32}(mapping=Dense(2, 4), σ2=4-element Array{Flo...)
 discriminator = (CMeanGaussian{Float32}(mapping=Dense(4, 1, σ), σ2=1-element Array...)
```
"""
struct GAN{T} <: AbstractGAN{T}
	prior::AbstractPDF
	generator::AbstractCPDF
	discriminator::AbstractCPDF
end

Flux.@functor GAN

GAN(p::AbstractPDF{T}, g::AbstractCPDF{T}, d::AbstractCPDF{T}) where T = GAN{T}(p, g, d)

function GAN(zlength::Int, g::AbstractCPDF{T}, d::AbstractCPDF{T}) where T
    μ = NoGradArray(zeros(T, zlength))
    σ = NoGradArray(ones(T, zlength))
    prior = Gaussian(μ, σ)
    GAN{T}(prior, g, d)
end

"""
	generator_loss(m::GAN, z::AbstractArray)
	generator_loss(m::GAN, batchsize::Int)

Loss of the GAN generator. The input is either the random code `z` or `batchsize` 
of samples to generate from the model prior and compute the loss from.
"""
generator_loss(m::GAN{T}, z::AbstractArray) where T = generator_loss(T,freeze(m.discriminator.mapping)(mean(m.generator,z)))
generator_loss(m::GAN{T}, batchsize::Int) where T = generator_loss(m, rand(m.prior, batchsize))


"""
	discriminator_loss(m::GAN, x::AbstractArray[, z::AbstractArray])

Loss of the GAN discriminator given a batch of training samples `x` and latent prior samples `z`.
If z is not given, it is automatically generated from the model prior.
"""
discriminator_loss(m::GAN{T}, x::AbstractArray, z::AbstractArray) where T = discriminator_loss(T, mean(m.discriminator,x), mean(m.discriminator, freeze(m.generator.mapping)(z)))
discriminator_loss(m::GAN{T}, x::AbstractArray) where T = discriminator_loss(m, x, rand(m.prior, size(x,2)))

function Base.show(io::IO, m::AbstractGAN{T}) where T
    p = repr(m.prior)
    p = sizeof(p)>70 ? "($(p[1:70-3])...)" : p
    g = repr(m.generator)
    g = sizeof(g)>70 ? "($(g[1:70-3])...)" : g
    d = repr(m.discriminator)
    d = sizeof(d)>70 ? "($(d[1:70-3])...)" : d
    msg = """$(typeof(m)):
     prior   = $(p)
     generator = $(g)
     discriminator = $(d)
    """
    print(io, msg)
end
