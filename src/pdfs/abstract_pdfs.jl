export mean, variance, mean_var, rand, loglikelihood, kld

abstract type AbstractPDF{T<:Real} end
abstract type AbstractCPDF{T<:Real} end

mean(p::AbstractPDF) = error("Not implemented!")
variance(p::AbstractPDF) = error("Not implemented!")
mean_var(p::AbstractPDF) = error("Not implemented!")
rand(p::AbstractCPDF; batch=1) = error("Not, implemented!")
loglikelihood(p::AbstractPDF, x::AbstractArray) = error("Not implemented!")

mean(p::AbstractCPDF, z::AbstractArray) = error("Not implemented!")
variance(p::AbstractCPDF, z::AbstractArray) = error("Not implemented!")
mean_var(p::AbstractCPDF, z::AbstractArray) = error("Not implemented!")
rand(p::AbstractCPDF, z::AbstractArray; batch=1) = error("Not, implemented!")
loglikelihood(p::AbstractCPDF, x::AbstractArray, z::AbstractArray) = error("Not implemented!")

kld(p::AbstractPDF, q::AbstractPDF) = error("Not implemented!")
kld(p::AbstractCPDF, q::AbstractPDF, z::AbstractArray) = error("Not implemented!")
