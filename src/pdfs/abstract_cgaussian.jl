export loglikelihood, kld, rand, xlength, zlength

abstract type AbstractCGaussian{T} <: AbstractCPDF{T} end

xlength(p::AbstractCGaussian) = p.xlength
zlength(p::AbstractCGaussian) = p.zlength

function rand(p::AbstractCGaussian{T}, z::AbstractArray; batch=1) where T
    (μ, σ2) = mean_var(p, z)
    k = xlength(p)
    μ .+ sqrt.(σ2) .* randn(T, k, batch)
end

function loglikelihood(p::AbstractCGaussian, x::AbstractArray, z::AbstractArray)
    (μ, σ2) = mean_var(p, z)
    k = xlength(p)
    - (sum((x - μ).^2 ./ σ2, dims=1) .+ sum(log.(σ2)) .+ k*log(2π)) ./ 2
end

function kld(p::AbstractCGaussian{T}, q::Gaussian{T}, z::AbstractArray) where T
    (μ1, σ1) = mean_var(p, z)
    (μ2, σ2) = mean_var(q)
    k = xlength(p)
    (-k + sum(log.(σ2 ./ σ1)) + sum(σ1 ./ σ2) .+ sum((μ2 .- μ1).^2 ./ σ1, dims=1)) ./ 2
end
