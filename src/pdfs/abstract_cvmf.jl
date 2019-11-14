export loglikelihood, kld, rand, mean_conc, concentration

abstract type AbstractCVMF{T} <: AbstractCPDF{T} end

function rand(p::AbstractCVMF{T}, z::AbstractArray{T}) where {T}
    (μ, κ) = mean_conc(p, z)
    sample_vmf(μ, κ)
end

function loglikelihood(p::AbstractCVMF{T}, x::AbstractArray{T}, z::AbstractArray{T}) where {T}
    (μ, κ) = mean_conc(p, z)
    log_vmf(x, μ, κ)
end

# This is here because we always compute KLD with VMF and hyperspherical uniform - nothing else as KLD between two VMFs is rather complicated to compute
"""
kld(p::AbstractCVMF, q::HypersphericalUniform, z::AbstractArray)

Compute Kullback-Leibler divergence between a conditional Von Mises-Fisher distribution `p` given `z`
and a hyperspherical uniform distribution `q` with the same dimensionality.
"""
function kld(p::AbstractCVMF{T}, q::HypersphericalUniform{T}, z::AbstractArray{T}) where {T}
    (μ, κ) = mean_conc(p, z)
    if size(μ, 1) != q.dims
        error("Cannot compute KLD between VMF and HSU with different dimensionality")
    end
    .- vmfentropy.(q.dims, κ) .+ huentropy(q.dims)
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




