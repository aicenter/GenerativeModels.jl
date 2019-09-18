export GAN

"""
	GAN

The Generative Adversarial Network.
"""
struct GAN{T} <: AbstractGAN{T}
	prior::AbstractPDF
	generator::AbstractCPDF
	discriminator::AbstractCPDF

	function GAN{T}(p::AbstractPDF{T}, g::AbstractCPDF{T}, d::AbstractCPDF{T}) where T
        if xlength(g) == zlength(d)
            new(p, g, d)
        else
            error("Encoder and decoder must have same zlength.")
        end
    end
end

Flux.@treelike GAN

function GAN(g::CGaussian{T}, d::CGaussian{T}) where T
    zlen = zlength(g)
    prior = Gaussian(zeros(T, zlen), ones(T, zlen))
    GAN{T}(prior, g, d)
end

generator_loss(m::GAN{T}, z::AbstractArray) where T = generator_loss(T,freeze(model.discriminator.mapping)(mean(m.generator,z)))
generator_loss(m::GAN{T}, batchsize::Int) where T = generator_loss(m, rand(m.prior, batchsize))

discriminator_loss(m::GAN{T}, x::AbstractArray, z::AbstractArray) where T = discriminator_loss(T, mean(m.discriminator,x), mean(m.discriminator, freeze(model.generator.mapping)(z)))
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


