export loglikelihood, kld, rand, xlength, zlength
export AbstractVar, DiagVar, ScalarVar, UnitVar

abstract type AbstractCGaussian{T} <: AbstractCPDF{T} end

"""Abstract variance type"""
abstract type AbstractVar end

"""Diagonal variance represented as a vector"""
struct DiagVar <: AbstractVar end

"""Scalar variance represented as a one-element vector"""
struct ScalarVar <: AbstractVar end

"""Unit variance represented by a vector of ones"""
struct UnitVar <: AbstractVar end


xlength(p::AbstractCGaussian) = p.xlength
zlength(p::AbstractCGaussian) = p.zlength

function rand(p::AbstractCGaussian{T}, z::AbstractArray) where T
    (μ, σ2) = mean_var(p, z)
    r = randn!(similar(μ))
    μ .+ sqrt.(σ2) .* r 
end

function loglikelihood(p::AbstractCGaussian{T}, x::AbstractArray, z::AbstractArray) where T
    (μ, σ2) = mean_var(p, z)
    d = x - μ
    y = d .* d
    y = (1 ./ σ2) .* y
    -sum(y, dims=1)
end

function kld(p::AbstractCGaussian{T}, q::Gaussian{T}, z::AbstractArray) where T
    N = size(z, 2)
    (μ1, σ1) = mean_var(p, z)
    (μ2, σ2) = mean_var(q)
    m1 = mean(log.(σ2 ./ σ1), dims=1)
    m2 = mean(σ1 ./ σ2, dims=1)
    d  = μ2 .- μ1
    dd = d .* d
    m3 = mean(dd ./ σ2, dims=1)
    m1 .+ m2 .+ m3
end



function detect_mapping_variant(x::AbstractVector, xlength::Int) where T
    if size(x) == (xlength,)
        return UnitVar
    elseif size(x) == (xlength+1,)
        return ScalarVar
    elseif size(x) == (xlength*2,)
        return DiagVar
    else
        error("Mapping output could not be matched with any variance type.")
    end
end

function detect_mapping_variant(x::AbstractMatrix, xlength::Int) where T
    detect_mapping_variant(x[:,1], xlength)
end

function detect_mapping_variant(mapping, xlength::Int, zlength::Int)
    p = first(params(mapping))
    z = randn(zlength, 1)
    z = isa(p, CuArray) ? gpu(z) : z
    x = mapping(z)
    detect_mapping_variant(x, xlength)
end

function detect_mapping_variant(mapping::Function, T::Type, xlength::Int, zlength::Int)
    z = randn(T, zlength, 1)
    x = mapping(z)
    detect_mapping_variant(x, xlength)
end
