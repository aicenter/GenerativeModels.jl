export loglikelihood, kld, rand, xlength, zlength

abstract type AbstractCGaussian{T} <: AbstractCPDF{T} end

xlength(p::AbstractCGaussian) = p.xlength
zlength(p::AbstractCGaussian) = p.zlength

function rand(p::AbstractCGaussian{T}, z::AbstractArray) where T
    (μ, σ2) = mean_var(p, z)
    r = randn!(similar(μ))
    μ .+ sqrt.(σ2) .* r 
end

function loglikelihood(p::AbstractCGaussian{T}, x::AbstractArray, z::AbstractArray) where T
    (μ, σ2) = mean_var(p, z)
    y = (x - μ).^2
    y = (1 ./ σ2) .* y
    -sum(y, dims=1)
end

function kld(p::AbstractCGaussian{T}, q::Gaussian{T}, z::AbstractArray) where T
    N = size(z, 2)
    (μ1, σ1) = mean_var(p, z)
    (μ2, σ2) = mean_var(q)
    mean(log.(σ2 ./ σ1), dims=1) .+ mean(σ1 ./ σ2, dims=1) .+ mean((μ2 .- μ1).^2 ./ σ2, dims=1)
end
