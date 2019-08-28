export VAE
export encoder_mean, encoder_variance, encoder_mean_var, encoder_sample, encoder_loglikelihood
export prior_mean, prior_variance, prior_mean_var, prior_sample, prior_loglikelihood
export decoder_mean, decoder_variance, decoder_mean_var, decoder_sample, decoder_loglikelihood
export elbo

struct VAE{T<:Real} <: AbstractAutoEncoder
    xsize::Int
    zsize::Int
    encoder
    decoder
    σz::Tracker.TrackedArray{T,1}
    σe::Tracker.TrackedArray{T,1}
end

Flux.@treelike VAE

"""`VAE(xsize, zsize, encoder, decoder)`

AutoEncoder that enforces sparsity on the latent layer.
p(x|z) = N(x|z, σe);
p(z)   = N(z|0, σz);
σz: point estimate
"""
function VAE{T}(xsize::Int, zsize::Int, encoder, decoder) where T
    σz = param(ones(T, zsize) .* 0.001f0)
    σe = param(ones(T, 1))
    VAE(xsize, zsize, encoder, decoder, σz, σe)
end


encoder_mean(m::VAE, x::AbstractArray) = m.encoder(x)
encoder_variance(m::VAE, x::AbstractArray) = m.σz
encoder_mean_var(m::VAE, x::AbstractArray) = (m.encoder(x), m.σz)

function encoder_sample(m::VAE{T}, x::AbstractArray) where T
    (μz, σz) = encoder_mean_var(m, x)
    μz .+ σz .* randn(T, m.zsize)
end

encoder_loglikelihood(m::VAE, z::AbstractArray) = error("Not implemented.")


prior_mean(m::VAE) = UniformScaling(0)
prior_variance(m::VAE) = I
prior_mean_var(m::VAE) = (UniformScaling(0), I)
prior_sample(m::VAE{T}) where T = randn(T, m.zsize)
prior_loglikelihood(m::VAE) = error("Not implemented.")


decoder_mean(m::VAE, z::AbstractArray) = m.decoder(z)
decoder_variance(m::VAE, z::AbstractArray) = m.σe
decoder_mean_var(m::VAE, z::AbstractArray) = (m.decoder(z), m.σe)

function decoder_sample(m::VAE, z::AbstractArray)
    (μx, σe) = decoder_mean_var(m, z)
    μx .+ σe .* randn(T, m.xsize)
end

function decoder_loglikelihood(m::VAE, x::AbstractArray, z::AbstractArray)
    xrec = decoder_mean(m, z)
    llh  = sum(abs2, x - xrec) / m.σe[1]^2 # TODO: what if σe is not scalar?
end


"""`elbo(m::VAE, x::AbstractArray)`

Computes variational lower bound.
"""
function elbo(m::VAE{T}, x::AbstractArray) where T
    N  = size(x, 2)
    σe = m.σe[1] # TODO: what if σe is not scalar?

    (μz, σz) = encoder_mean_var(m, x)
    # z = encoder_sample(m, x) TODO: this would recompute μz, σz ... change interface of encoder_sample?
    z = μz .+ σz .* randn(T, m.zsize)

    llh = decoder_loglikelihood(m, x, z) / N
    KLz = (sum(μz.^2 .+ σz.^2) / 2. - sum(log.(abs.(σz)))) / N

    loss = llh + KLz + log(σe)*m.zsize/2
end


function Base.show(io::IO, m::VAE{T}) where T
    e = summary(m.encoder)
    e = sizeof(e)>80 ? e[1:77]*"..." : e
    d = summary(m.decoder)
    d = sizeof(d)>80 ? d[1:77]*"..." : d
    msg = """VAE{$T}:
      xsize   = $(m.xsize)
      zsize   = $(m.zsize)
      encoder = $(e)
      decoder = $(d)
      σz      = $(m.σz)
      σe      = $(m.σe)
    """
    print(io, msg)
end
