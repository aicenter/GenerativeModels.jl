export mean, variance, mean_var, rand, loglikelihood, kld, mmd
export length, xlength, zlength

abstract type AbstractPDF{T<:Real} end
abstract type AbstractCPDF{T<:Real} end

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
rand(p::AbstractPDF, batchsize::Int=1) = error("Not, implemented!")

"""
    loglikelihood(p::AbstractPDF, x::AbstractArray)

Computes log p(x|μ,σ2).
"""
loglikelihood(p::AbstractPDF, x::AbstractArray) = error("Not implemented!")


"""
    length(p::AbstractPDF)

Returns the length of a sample.
"""
length(p::AbstractPDF) = error("Not implemented!")


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
    rand(p::AbstractCPDF, z::AbstractArray)

Produce `batch` number of samples from a conditional PDF.
"""
rand(p::AbstractCPDF, z::AbstractArray) = error("Not, implemented!")

"""
    loglikelihood(p::AbstractCPDF, x::AbstractArray, z::AbstractArray)

Computes log p(x|z).
"""
loglikelihood(p::AbstractCPDF, x::AbstractArray, z::AbstractArray) = error("Not implemented!")

"""
    xlength(p::AbstractCPDF)

Returns the length of a sample in data space.
"""
xlength(p::AbstractCPDF) = error("Not implemented!")

"""
    zlength(p::AbstractCPDF)

Returns the length of a sample in latent space.
"""
zlength(p::AbstractCPDF) = error("Not implemented!")



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

"""
    mmd(p::AbstractCPDF, q::AbstractPDF, z::AbstractArray, k)

Compute the maximum mean discrepancy between a conditional PDF `p` given `z` 
and a PDF `q`, given kernel `k`.
"""
mmd(p::AbstractCPDF, q::AbstractPDF, z::AbstractArray, k) = mmd(k, rand(p,z), rand(q, size(z,2)))

function _trainable(m)
    ps = Flux.functor(m)[1]
    (; [k=>ps[k] for k in keys(ps) if !isa(ps[k], NoGradArray)]...)
end

"""Custom function that ignores `NoGradArray`s"""
Flux.trainable(m::AbstractCPDF) = _trainable(m)
"""Custom function that ignores `NoGradArray`s"""
Flux.trainable(m::AbstractPDF) = _trainable(m)
