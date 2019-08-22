export ARDAutoEncoder
export elbo, loglikelihood, loglatent

struct ARDAutoEncoder{T<:Real} <: AbstractAutoEncoder
    xsize::Int
    zsize::Int
    encoder
    decoder
    γ::Tracker.TrackedArray{T,1}
end

Flux.@treelike ARDAutoEncoder

"""`ARDAutoEncoder(encoder, decoder)`

AutoEncoder that enforces sparsity on the latent layer.
"""
function ARDAutoEncoder{T}(xsize::Int, zsize::Int, encoder, decoder) where T
    γ = param(ones(T, zsize))
    ARDAutoEncoder(xsize, zsize, encoder, decoder, γ)
end



"""`encoder_params(m::ARDAutoEncoder, x::AbstractArray)`

Return the parameters of the latent distribution.
"""
function encoder_params(m::ARDAutoEncoder, x::AbstractArray)
    z = m.encoder(x)
    μz = z[1:m.zsize]
    σz = z[m.zsize:end]
    (μz, σz, m.γ)
end


"""`encoder_sample(m::ARDAutoEncoder, x::AbstractArray)`

Sample from the latent layer.
"""
function encoder_sample(m::ARDAutoEncoder{T}, x::AbstractArray) where T
    (μz, σz, γ) = encoder_params(m, x)
    γ .* (μz + σz .* randn(T, m.zsize))
end


"""`decode(m::ARDAutoEncoder, z::AbstractArray)`

Reconstruct from a (sampled) latent vector
"""
function decode(m::ARDAutoEncoder, z::AbstractArray)
    m.decoder(z)
end


"""`elbo(m::ARDAutoEncoder, x::AbstractArray)`

Computes variational lower bound.
"""
function elbo(m::ARDAutoEncoder, x::AbstractArray)
    error("Not implemented")
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
