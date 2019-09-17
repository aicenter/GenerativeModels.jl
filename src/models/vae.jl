export VAE
export elbo, mmd

"""
    VAE{T}(prior::Gaussian, encoder::CGaussian, decoder::CGaussian)

Variational Auto-Encoder.

# Example
Create a vanilla VAE with standard normal prior with:
```julia-repl
julia> enc = CGaussian{Float32,DiagVar}(2,5,Dense(5,4))
CGaussian{Float32,DiagVar}(xlength=2, zlength=5, mapping=Dense(5, 4))

julia> dec = CGaussian{Float32,ScalarVar}(5,2,Dense(2,6))
CGaussian{Float32,ScalarVar}(xlength=5, zlength=2, mapping=Dense(2, 6))

julia> vae = VAE(enc, dec)
VAE{Float32}:
  prior   = Gaussian{Float32}(μ=2-element Array{Float32,1}, σ2=2-element Array{Float32,1})
  encoder = CGaussian{Float32,DiagVar}(xlength=2, zlength=5, mapping=Dense(5, 4))
  decoder = CGaussian{Float32,ScalarVar}(xlength=5, zlength=2, mapping=Dense(2, 6))
```
"""
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

function elbo(m::AbstractVAE, x::AbstractArray; β=1)
    z = rand(m.encoder, x)
    llh = mean(-loglikelihood(m.decoder, x, z))
    kl  = mean(kld(m.encoder, m.prior, x))
    llh + β*kl
end

mmd(m::AbstractVAE, x::AbstractArray, k) = mmd(m.encoder, m.prior, x, k)

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
