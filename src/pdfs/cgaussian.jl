export CGaussian
export XVar, DiagVar, ScalarVar, UnitVar
export mean, variance, mean_var, sample, loglikelihood, kld

abstract type XVar end
struct DiagVar <: XVar end
struct ScalarVar <: XVar end
struct UnitVar <: XVar end

struct CGaussian{T,V<:XVar} <: AbstractCPDF{T}
    xlength::Int
    zlength::Int
    encoder

    function CGaussian{T,V}(xlength, zlength, encoder) where T where V

        if T == Float32
            encoder = f32(encoder)
        elseif T == Float64
            encoder = f64(encoder)
        else
            error("Encoder cannot be converted to type $T")
        end

        cg = new(xlength, zlength, encoder)
        ex = encoder(randn(T, zlength))

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
    return ex[1:p.xlength,:], softplus.(ex[p.xlength+1:end,:])
end

function mean_var(p::CGaussian{T,UnitVar}, z::AbstractArray) where T
    μ = p.encoder(z)
    return μ, ones(T, xlength(p))
end

mean(p::CGaussian, z::AbstractArray) = mean_var(p, z)[1]
variance(p::CGaussian, z::AbstractArray) = mean_var(p, z)[2]

function sample(p::CGaussian{T}, z::AbstractArray; batch=1) where T
    (μ, σ2) = mean_var(p, z)
    k = xlength(p)
    μ .+ sqrt.(σ2) .* randn(T, k, batch)
end

function loglikelihood(p::CGaussian, x::AbstractArray, z::AbstractArray)
    (μ, σ2) = mean_var(p, z)
    k = xlength(p)
    - (sum((x - μ).^2 ./ σ2, dims=1) .+ sum(log.(σ2)) .+ k*log(2π)) ./ 2
end

function kld(p::CGaussian{T}, q::Gaussian{T}, z::AbstractArray) where T
    (μ1, σ1) = mean_var(p, z)
    (μ2, σ2) = mean_var(q)
    k = xlength(p)
    (-k + sum(log.(σ2 ./ σ1)) + sum(σ1 ./ σ2) .+ sum((μ2 .- μ1).^2 ./ σ1, dims=1)) ./ 2
end

function Base.show(io::IO, p::CGaussian{T,V}) where T where V
    e = repr(p.encoder)
    e = sizeof(e)>50 ? "($(e[1:47])...)" : e
    msg = "CGaussian{$T,$V}(xlength=$(p.xlength), zlength=$(p.zlength), encoder=$e)"
    print(io, msg)
end
