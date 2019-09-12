export mean, variance, mean_var, rand, loglikelihood, kld

struct SharedVarCGaussian{T} <: AbstractCPDF{T}
    xlength::Int
    zlength::Int
    mapping
    σ2::AbstractArray{T}

    function SharedVarCGaussian{T}(xlength, zlength, mapping, σ2) where T

        if T == Float32
            mapping = f32(mapping)
        elseif T == Float64
            mapping = f64(mapping)
        else
            error("Encoder cannot be converted to type $T")
        end

        cg = new(xlength, zlength, mapping, σ2)
        ex = mapping(randn(T, zlength))
        size(ex) == (xlength,) ? cg : error("Mapping must return samples of xlength")
    end

end

mean_var(p::SharedVarCGaussian, z::AbstractArray) = (p.mapping(z), p.σ2)

function rand(p::SharedVarCGaussian, z::AbstractArray; batch=1)
    (μ, σ2) = mean_var(p, z)
    k = xlength(p)
    μ .+ sqrt.(σ2) .* randn(T, k, batch)
end

function loglikelihood(p::SharedVarCGaussian, x::AbstractArray, z::AbstractArray)
    (μ, σ2) = mean_var(p, z)
    k = xlength(p)
    - (sum((x - μ).^2 ./ σ2, dims=1) .+ sum(log.(σ2)) .+ k*log(2π)) ./ 2
end

function kld(p::SharedVarCGaussian{T}, q::Gaussian{T}, z::AbstractArray) where T
    (μ1, σ1) = mean_var(p, z)
    (μ2, σ2) = mean_var(q)
    k = xlength(p)
    (-k + sum(log.(σ2 ./ σ1)) + sum(σ1 ./ σ2) .+ sum((μ2 .- μ1).^2 ./ σ1, dims=1)) ./ 2
end

function Base.show(io::IO, p::SharedVarCGaussian{T,V}) where T where V
    e = repr(p.mapping)
    e = sizeof(e)>50 ? "($(e[1:47])...)" : e
    msg = "SharedVarCGaussian{$T,$V}(xlength=$(p.xlength), zlength=$(p.zlength), mapping=$e, σ2=$(summary(p.σ2))"
    print(io, msg)
end
