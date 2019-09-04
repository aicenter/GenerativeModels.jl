export encoder_mean, encoder_variance, encoder_mean_var
export encoder_sample, encoder_loglikelihood
export prior_loglikelihood
export decoder_mean, decoder_variance, decoder_mean_var
export decoder_sample, decoder_loglikelihood

abstract type AbstractVAE{T<:Real} <: AbstractGM end

encoder_mean(m::AbstractVAE, x::AbstractArray) = m.encoder(x)
encoder_variance(m::AbstractVAE, x::AbstractArray) = m.σz.^2
encoder_mean_var(m::AbstractVAE, x::AbstractArray) = (encoder_mean(m, x), encoder_variance(m, x))
encoder_loglikelihood(m::AbstractVAE, z::AbstractArray) = error("Not implemented.")

function encoder_sample(m::AbstractVAE{T}, μz::AbstractArray, σz::AbstractArray) where T
    μz .+ sqrt.(σz) .* randn(T, m.zsize)
end

function encoder_sample(m::AbstractVAE{T}, x::AbstractArray) where T
    (μz, σz) = encoder_mean_var(m, x)
    encoder_sample(m, μz, σz)
end


decoder_mean(m::AbstractVAE, z::AbstractArray) = m.decoder(z)
decoder_variance(m::AbstractVAE, z::AbstractArray) = m.σe.^2
decoder_mean_var(m::AbstractVAE, z::AbstractArray) = (decoder_mean(m, z), decoder_variance(m, z))

function decoder_loglikelihood(m::AbstractVAE, x::AbstractArray, z::AbstractArray)
    @assert size(x, 1) == m.xsize
    @assert size(z, 1) == m.zsize
    (μx, σe) = decoder_mean_var(m, z)
    -dropdims(sum((x - μx).^2 ./ σe, dims=1), dims=1)
end

function decoder_sample(m::AbstractVAE{T}, μx::AbstractArray, σe::AbstractArray) where T
    μx .+ sqrt.(σe) .* randn(T, m.xsize)
end

function decoder_sample(m::AbstractVAE{T}, z::AbstractArray) where T
    (μx, σe) = decoder_mean_var(m, z)
    decoder_sample(m, μx, σe)
end
