export loglikelihood, kld, rand, mean_conc, concentration

abstract type AbstractCVMF{T} <: AbstractCPDF{T} end

function rand(p::AbstractCVMF, z::AbstractArray)
    (μ, κ) = mean_conc(p, z)
    sample_vmf(μ, κ)
end

function loglikelihood(p::AbstractCVMF, x::AbstractArray, z::AbstractArray)
    (μ, κ) = mean_conc(p, z)
    log_vmf(x, μ, κ)
end

# This is here because we always compute KLD with VMF and hyperspherical uniform - nothing else as KLD between two VMFs is rather complicated to compute
"""
    kld(p::AbstractCVMF{T}, z::AbstractArray{T})

Compute Kullback-Leibler divergence between a conditional Von Mises-Fisher distribution `p` given `z`
and a hyperspherical uniform distribution with the same dimensionality
"""
function kld(p::AbstractCVMF, z::AbstractArray)
    dims = size(z, 1)
    .- vmfentropy(dims, concentration(p)) .+ huentropy(dims)
end

"""
    mean_conc(p::AbstractCVMF, z::AbstractArray)

Returns mean and concentration of a conditional VMF distribution.
"""
mean_conc(p::AbstractCVMF, z::AbstractArray) = error("Not implemented!")


"""
    mean(p::AbstractCVMF, z::AbstractArray)

Returns mean of a conditional VMF distribution.
"""
mean(p::AbstractCVMF, z::AbstractArray) = mean_conc(p, z)[1]

"""
    concentration(p::AbstractCVMF, z::AbstractArray)

Returns variance of a conditional VMF distribution.
"""
concentration(p::AbstractCVMF, z::AbstractArray) = mean_conc(p, z)[2]




