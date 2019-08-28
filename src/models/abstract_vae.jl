export encoder_mean, encoder_variance, encoder_mean_var, encoder_loglikelihood
export prior_loglikelihood
export decoder_mean, decoder_variance, decoder_mean_var, decoder_loglikelihood

abstract type AbstractVAE <: AbstractGN end

encoder_mean(m::AbstractVAE, x::AbstractArray) = m.encoder(x)
encoder_variance(m::AbstractVAE, x::AbstractArray) = abs.(m.σz)
encoder_mean_var(m::AbstractVAE, x::AbstractArray) = (m.encoder(x), abs.(m.σz))
encoder_loglikelihood(m::AbstractVAE, z::AbstractArray) = error("Not implemented.")

prior_loglikelihood(m::AbstractVAE) = error("Not implemented.")

decoder_mean(m::AbstractVAE, z::AbstractArray) = m.decoder(z)
decoder_variance(m::AbstractVAE, z::AbstractArray) = abs.(m.σe)
decoder_mean_var(m::AbstractVAE, z::AbstractArray) = (m.decoder(z), abs.(m.σe))

function decoder_loglikelihood(m::AbstractVAE, x::AbstractArray, z::AbstractArray)
    xrec = decoder_mean(m, z)
    llh  = sum(abs2, x - xrec) / m.σe[1]^2 # TODO: what if σe is not scalar?
end
