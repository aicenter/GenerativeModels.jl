export ARDAutoEncoder
export elbo, loglikelihood, loglatent

struct ARDAutoEncoder{T<:Real} <: AbstractAutoEncoder
    xsize::Int
    zsize::Int
    encoder
    decoder
    σz::Tracker.TrackedArray{T,1}
    γ::Tracker.TrackedArray{T,1}
    σe::Tracker.TrackedArray{T,1}
end

Flux.@treelike ARDAutoEncoder

"""`ARDAutoEncoder(xsize, zsize, encoder, decoder)`

AutoEncoder that enforces sparsity on the latent layer.
"""
function ARDAutoEncoder{T}(xsize::Int, zsize::Int, encoder, decoder) where T
    γ = param(ones(T, zsize) .* 0.001f0)
    σz = param(ones(T, zsize) .* 0.001f0)
    σe = param(ones(T, 1) .* 0.001f0)
    ARDAutoEncoder(xsize, zsize, encoder, decoder, σz, γ, σe)
end


"""`encoder_params(m::ARDAutoEncoder, x::AbstractArray)`

Return the parameters of the latent distribution.
"""
function encoder_params(m::ARDAutoEncoder, x::AbstractArray)
    μz = m.encoder(x)
    (μz, m.σz, m.γ)
end


"""`encoder_sample(m::ARDAutoEncoder, x::AbstractArray)`

Sample from the latent layer.
"""
function encoder_sample(m::ARDAutoEncoder{T}, ps) where T
    (μz, σz, γ) = ps
    γ .* (μz .+ σz .* randn(T, m.zsize))
end


"""`decode(m::ARDAutoEncoder, z::AbstractArray)`

Reconstruct from a (sampled) latent vector
"""
function decode(m::ARDAutoEncoder, z::AbstractArray)
    m.decoder(z)
end

function (m::ARDAutoEncoder)(x)
    (μz, _, γ) = encoder_params(m, x)
    decode(m, μz .* γ)
end


"""`elbo(m::ARDAutoEncoder, x::AbstractArray)`

Computes variational lower bound.
"""
function elbo(m::ARDAutoEncoder, x::AbstractArray)
    N  = size(x, 2)
    σe = m.σe[1]

    ps = encoder_params(m, x)
    (μz, σz, γ) = ps
    z = encoder_sample(m, ps)
    xrec = decode(m, z)

    llh = sum(abs2, x - xrec) ./ σe^2
    KLz = sum(μz.^2 .+ σz.^2) / 2. - sum(log.(abs.(σz)))

    loss = llh + KLz + σe^2*N*m.zsize
end


"""`loglikelihood(m::ARDAutoEncoder, x::AbstractArray, z::AbstractArray)'

Computes the log-likelihood of the data.
"""
function loglikelihood(m::ARDAutoEncoder, x::AbstractArray, z::AbstractArray)
    error("Not implemented")
end


"""`loglatent(m::ARDAutoEncoder, z::AbstractArray)`

Compute the probability of a latent vector.
"""
function loglatent(m::ARDAutoEncoder, z::AbstractArray)
    error("Not implemented")
end


function Base.show(io::IO, m::ARDAutoEncoder{T}) where T
    e = summary(m.encoder)
    e = sizeof(e)>80 ? e[1:77]*"..." : e
    d = summary(m.decoder)
    d = sizeof(d)>80 ? d[1:77]*"..." : d
    msg = """ARDAutoEncoder{$T}:
      xsize   = $(m.xsize)
      zsize   = $(m.zsize)
      encoder = $(e)
      decoder = $(d)
      σz      = $(m.σz)
      γ       = $(m.γ)
      σe      = $(m.σe)
    """
    print(io, msg)
end
