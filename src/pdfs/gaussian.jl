export Gaussian, CGaussian
export mean, variance, mean_var, sample, loglikelihood, kld

struct Gaussian{T} <: AbstractPDF{T}
    μ::AbstractArray{T}
    σ2::AbstractArray{T}
end

Flux.@treelike Gaussian


length(p::Gaussian) = size(p.μ, 1)
mean(p::Gaussian) = p.μ
variance(p::Gaussian) = p.σ2
mean_var(p::Gaussian) = (p.μ, p.σ2)

function sample(p::Gaussian{T}; batch=1) where T
    k = length(p)
    μ, σ2 = mean_var(p)
    μ .+ sqrt.(σ2) .* randn(T, k, batch)
end

function loglikelihood(p::Gaussian, x::AbstractArray)
    k = length(p)
    - (sum((x .- p.μ).^2 ./ p.σ2, dims=1) .+ sum(log.(p.σ2)) .+ k*log(2π)) ./ 2
end

abstract type XVar end
struct DiagVar <: XVar end
struct ScalarVar <: XVar end
struct UnitVar <: XVar end

struct CGaussian{T,V<:XVar} <: AbstractCPDF{T}
    xlength::Int
    zlength::Int
    encoder

    function CGaussian{T,V}(xlength, zlength, encoder) where T where V
        ex = encoder(randn(T, zlength))
        cg = new(xlength, zlength, encoder)
        if V == UnitVar
            size(ex) == (xlength,) ? cg : error("With UnitVar encoder must return samples of xlength")
        elseif V == ScalarVar
            size(ex) == (xlength+1,) ? cg : error("With ScalarVar encoder must return samples of xlength+1")
        else
            size(ex) == (xlength*2,) ? cg : error("With DiagVar encoder must return samples of xlength*2")
        end
    end
end

Flux.@treelike CGaussian


xlength(p::CGaussian) = p.xlength
zlength(p::CGaussian) = p.zlength
length(p::CGaussian) = (p.xlength, p.zlength) # TODO: how to deal with this?

function mean_var(p::CGaussian{T}, z::AbstractArray) where T
    ex = p.encoder(z)
    # TODO: replace with softplus_safe
    return ex[1:p.xlength,:], softplus.(ex[p.xlength+1:end])
end

function mean_var(p::CGaussian{T,UnitVar}, z::AbstractArray) where T
    μ = p.encoder(z)
    return μ, I
end

mean(p::CGaussian, z::AbstractArray) = mean_var(p, z)[1]
variance(p::CGaussian, z::AbstractArray) = mean_var(p, z)[2]

function sample(p::CGaussian{T}, z::AbstractArray; batch=1) where T
    (μ, σ2) = mean_var(p, z)
    k = xlength(p)
    μ .+ sqrt.(σ2) .* randn(T, k, batch)
end

function sample(p::CGaussian{T,UnitVar}, z::AbstractArray; batch=1) where T
    k = length(p)
    mean(p) .+ randn(T, k, batch)
end


function loglikelihood(p::CGaussian, x::AbstractArray, z::AbstractArray)
    (μ, σ2) = mean_var(p, z)
    k = xlength(p)
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


function Base.show(io::IO, p::Gaussian{T}) where T
    msg = "Gaussian{$T}(μ=$(summary(p.μ)), σ2=$(summary(p.σ2)))"
    print(io, msg)
end

function Base.show(io::IO, p::CGaussian{T,V}) where T where V
    e = repr(p.encoder)
    e = sizeof(e)>50 ? "($(e[1:47])...)" : e
    msg = "CGaussian{$T,$V}(xlength=$(p.xlength), zlength=$(p.zlength), encoder=$e)"
    print(io, msg)
end
