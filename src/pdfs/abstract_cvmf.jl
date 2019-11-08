export loglikelihood, kld, rand

abstract type AbstractCVMF{T} <: AbstractCPDF{T} end

function rand(p::AbstractCVMF{T}, z::AbstractArray{T}) where {T}
    (μ, κ) = mean_conc(p, z)
    #! Finish the sampling method for VMF
end

function loglikelihood(p::AbstractCVMF{T}, x::AbstractArray{T}, z::AbstractArray{T}) where {T}
    (μ, κ) = mean_conc(p, z)
    log_vmf(x, μ, κ)
end

# This is here because we always compute KLD with VMF and hyperspherical uniform - nothing else
"""
    kld(p::AbstractCVMF{T}, z::AbstractArray{T})

Compute Kullback-Leibler divergence between a conditional Von Mises-Fisher distribution `p` given `z`
and a hyperspherical uniform distribution with the same dimensionality
"""
function kld(p::AbstractCVMF{T}, z::AbstractArray{T}) where {T}
    dims = size(z, 1)
    .- vmfentropy(dims, conc(p)) .+ huentropy(dims)
end




