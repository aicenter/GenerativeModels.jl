export VAE
export elbo

struct VAE{T} <: AbstractVAE{T}
    prior::Gaussian
    encoder::CGaussian
    decoder::CGaussian

    function VAE{T}(p::Gaussian{T}, e::CGaussian{T}, d::CGaussian{T}) where T
        if xlength(e) == zlength(d)
            new(p, e, d)
        else
            error("Encoder and decoder must have same zlength.")
        end
    end

end

Flux.@treelike VAE

function VAE(enc::CGaussian{T}, dec::CGaussian{T}) where T
    zlen = zlength(dec)
    prior = Gaussian(zeros(T, zlen), ones(T, zlen))
    VAE{T}(prior, enc, dec)
end

function elbo(m::VAE, x::AbstractArray; β=1)
    z = sample(m.encoder, x)
    llh = Statistics.mean(-loglikelihood(m.decoder, x, z))
    kl  = Statistics.mean(kld(m.encoder, m.prior, x))
    llh + β*kl
end



function Base.show(io::IO, m::VAE{T}) where T
    msg = """VAE{$T}:
      prior   = $(m.prior)
      encoder = $(m.encoder)
      decoder = $(m.decoder)
    """
    print(io, msg)
end
