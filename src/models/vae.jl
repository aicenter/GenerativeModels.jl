export VAE
export elbo

struct VAE{T} <: AbstractVAE{T}
    xsize::Int
    zsize::Int
    prior::Gaussian
    encoder::CGaussian
    decoder::CGaussian
end

Flux.@treelike VAE

function VAE(xsize::Int, zsize::Int, enc::CGaussian{T}, dec::CGaussian{T}) where T
    prior = Gaussian(zeros(T, zsize), ones(T, zsize))
    VAE{T}(xsize, zsize, prior, enc, dec)
end

function elbo(m::VAE, x::AbstractArray; β=1)
    z = sample(m.encoder, x)
    llh = Statistics.mean(-loglikelihood(m.decoder, x, z))
    kl  = Statistics.mean(kld(m.encoder, m.prior, x))
    llh + β*kl
end



function Base.show(io::IO, m::VAE{T}) where T
    msg = """VAE{$T}:
      xsize   = $(m.xsize)
      zsize   = $(m.zsize)
      prior   = $(m.prior)
      encoder = $(m.encoder)
      decoder = $(m.decoder)
    """
    print(io, msg)
end
