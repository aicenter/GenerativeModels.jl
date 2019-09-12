export Gaussian
export mean, variance, mean_var, rand, loglikelihood, kld, length

struct Gaussian{T} <: AbstractPDF{T}
    μ::AbstractArray{T}
    σ2::AbstractArray{T}
end

Flux.@treelike Gaussian


length(p::Gaussian) = size(p.μ, 1)
mean(p::Gaussian) = p.μ
variance(p::Gaussian) = p.σ2
mean_var(p::Gaussian) = (p.μ, p.σ2)

function rand(p::Gaussian{T}; batch=1) where T
    k = length(p)
    μ, σ2 = mean_var(p)
    μ .+ sqrt.(σ2) .* randn(T, k, batch)
end

function loglikelihood(p::Gaussian, x::AbstractArray)
    k = length(p)
    - (sum((x .- p.μ).^2 ./ p.σ2, dims=1) .+ sum(log.(p.σ2)) .+ k*log(2π)) ./ 2
end

function kld(p::Gaussian, q::Gaussian)
    (μ1, σ1) = mean_var(p)
    (μ2, σ2) = mean_var(q)
    k = length(p)
    (-k + sum(log.(σ2 ./ σ1)) + sum(σ1 ./ σ2) .+ sum((μ2 - μ1).^2 ./ σ1, dims=1)) ./ 2
end

function Base.show(io::IO, p::Gaussian{T}) where T
    msg = "Gaussian{$T}(μ=$(summary(p.μ)), σ2=$(summary(p.σ2)))"
    print(io, msg)
end
