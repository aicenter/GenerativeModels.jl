export mean, variance, mean_var, rand, loglikelihood, kld
export AbstractVar, DiagVar, ScalarVar, UnitVar

abstract type AbstractPDF{T<:Real} end
abstract type AbstractCPDF{T<:Real} end

"""Abstract variance type"""
abstract type AbstractVar end

"""Diagonal variance represented as a vector"""
struct DiagVar <: AbstractVar end

"""Scalar variance represented as a one-element vector"""
struct ScalarVar <: AbstractVar end

"""Unit variance represented by a vector of ones"""
struct UnitVar <: AbstractVar end



"""
    mean_var(p::AbstractPDF)

Returns mean and variance of a PDF.
"""
mean_var(p::AbstractPDF) = error("Not implemented!")

"""
    mean(p::AbstractPDF)

Returns mean of a PDF.
"""
mean(p::AbstractPDF) = mean_var(p::AbstractPDF)[1]

"""
    variance(p::AbstractPDF)

Returns variance of a PDF.
"""
variance(p::AbstractPDF) = mean_var(p::AbstractPDF)[2]

"""
    rand(p::AbstractPDF; batch=1)

Produce `batch` number of samples from a PDF.
"""
rand(p::AbstractPDF; batch=1) = error("Not, implemented!")

"""
    loglikelihood(p::AbstractCPDF, x::AbstractArray)

Computes log p(x|μ,σ2).
"""
loglikelihood(p::AbstractPDF, x::AbstractArray) = error("Not implemented!")



"""
    mean_var(p::AbstractCPDF, z::AbstractArray)

Returns mean and variance of a conditional PDF.
"""
mean_var(p::AbstractCPDF, z::AbstractArray) = error("Not implemented!")


"""
    mean(p::AbstractCPDF, z::AbstractArray)

Returns mean of a conditional PDF.
"""
mean(p::AbstractCPDF, z::AbstractArray) = mean_var(p, z)[1]

"""
    variance(p::AbstractCPDF, z::AbstractArray)

Returns variance of a conditional PDF.
"""
variance(p::AbstractCPDF, z::AbstractArray) = mean_var(p, z)[2]

"""
    rand(p::AbstractCPDF, z::AbstractArray; batch=1)

Produce `batch` number of samples from a conditional PDF.
"""
rand(p::AbstractCPDF, z::AbstractArray; batch=1) = error("Not, implemented!")

"""
    loglikelihood(p::AbstractCPDF, x::AbstractArray, z::AbstractArray)

Computes log p(x|z).
"""
loglikelihood(p::AbstractCPDF, x::AbstractArray, z::AbstractArray) = error("Not implemented!")



"""
    kld(p::AbstractPDF, q::AbstractPDF, z::AbstractArray)

Compute Kullback-Leibler divergence between two PDFs.
"""
kld(p::AbstractPDF, q::AbstractPDF) = error("Not implemented!")

"""
    kld(p::AbstractCPDF, q::AbstractPDF, z::AbstractArray)

Compute Kullback-Leibler divergence between a conditional PDF `p` given `z`
and a PDF `q`.
"""
kld(p::AbstractCPDF, q::AbstractPDF, z::AbstractArray) = error("Not implemented!")
