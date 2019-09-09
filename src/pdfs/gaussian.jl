export Gaussian, CGaussian
export mean, variance, mean_var, sample, loglikelihood, kld


struct Gaussian{T} <: AbstractPDF{T}
    μ::AbstractArray{T}
    σ2::AbstractArray{T}
end

length(p::Gaussian) = size(p.μ, 1)
mean(p::Gaussian) = p.μ
variance(p::Gaussian) = p.σ2
mean_var(p::Gaussian) = (p.μ, p.σ2)

function sample(p::Gaussian{T}; batch=1) where T
    k = length(p)
    p.μ .+ sqrt.(p.σ2) .* randn(T, k, batch)
end

function loglikelihood(p::Gaussian, x::AbstractArray)
    k = length(p)
    - (sum((x .- p.μ).^2 ./ p.σ2, dims=1) .+ sum(log.(p.σ2)) .+ k*log(2π)) ./ 2
end



struct CGaussian{T} <: AbstractCPDF{T}
    μ
    σ2::AbstractArray{T}
end

length(p::CGaussian) = size(p.σ2, 1)
mean(p::CGaussian, z::AbstractArray) = p.μ(z)
variance(p::CGaussian, z::AbstractArray) = p.σ2
mean_var(p::CGaussian, z::AbstractArray) = (mean(p, z), variance(p, z))

function sample(p::CGaussian{T}, z::AbstractArray; batch=1) where T
    (μ, σ2) = mean_var(p, z)
    k = length(p)
    μ .+ sqrt.(σ2) .* randn(T, k, batch)
end

function loglikelihood(p::CGaussian, x::AbstractArray, z::AbstractArray)
    (μ, σ2) = mean_var(p, z)
    k = length(p)
    - (sum((x - μ).^2 ./ σ2, dims=1) .+ sum(log.(σ2)) .+ k*log(2π)) ./ 2
end



function kld(p::Gaussian, q::Gaussian)
    (μ1, σ1) = mean_var(p)
    (μ2, σ2) = mean_var(q)
    k = length(p)
    (-k + sum(log.(σ2 ./ σ1)) + sum(σ1 ./ σ2) .+ sum((μ2 - μ1).^2 ./ σ1, dims=1)) ./ 2
end

function kld(p::CGaussian, q::Gaussian, z::AbstractArray)
    (μ1, σ1) = mean_var(p, z)
    (μ2, σ2) = mean_var(q)
    k = length(p)
    (-k + sum(log.(σ2 ./ σ1)) + sum(σ1 ./ σ2) .+ sum((μ2 .- μ1).^2 ./ σ1, dims=1)) ./ 2
end


Flux.@treelike Gaussian
Flux.@treelike CGaussian

function Base.show(io::IO, p::Gaussian{T}) where T
    msg = "Gaussian{$T}(μ=$(summary(p.μ)), σ2=$(summary(p.σ2)))"
    print(io, msg)
end

function Base.show(io::IO, p::CGaussian{T}) where T
    μ = repr(p.μ)
    μ = sizeof(μ)>50 ? "($(μ[1:47])...)" : μ
    msg = "Gaussian{$T}(μ=$(μ), σ2=$(summary(p.σ2)))"
    print(io, msg)
end
