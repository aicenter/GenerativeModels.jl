export CGaussian
export mean_var

"""
    CGaussian{T,AbstractVar}

Conditional Gaussian that maps an input of `zlength` to its mean of `xlength`.
The mapping must output dimensions appropriate for the chosen variance type

# Arguments
- `xlength::Int`: length of mean
- `zlength::Int`: length of condition
- `mapping`: maps condition z to μ=mapping(z) (e.g. a Flux Chain)
- `AbstractVar`: one of the variance types: DiagVar, ScalarVar, UnitVar

# Example
```julia-repl
julia> p = CGaussian(3, 2, Dense(2, 3))
CGaussian{Float64,UnitVar}(xlength=3, zlength=2, mapping=Dense(2, 3))

julia> mean_var(p, ones(2))
([-0.339991, -0.061213, -0.769473] (tracked), [1.0, 1.0, 1.0])

julia> rand(p, ones(2))
Tracked 3×1 Array{Float64,2}:
 0.1829532154926673
 0.1235498922955946
 0.0767166501426535
```
"""
struct CGaussian{T,V<:AbstractVar} <: AbstractCGaussian{T}
    xlength::Int
    zlength::Int
    mapping
end

# function CGaussian(xlength::Int, zlength::Int, mapping::Function, T=Float32)
#     V = detect_mapping_variant(mapping, T, xlength, zlength)
#     CGaussian{T,V}(xlength, zlength, mapping)
# end
# 
# function CGaussian(xlength::Int, zlength::Int, mapping)
#     T = eltype(first(params(mapping)))
#     V = detect_mapping_variant(mapping, xlength, zlength)
#     CGaussian{T,V}(xlength, zlength, mapping)
# end

Flux.@functor CGaussian

function mean_var(p::CGaussian{T}, z::AbstractArray) where T
    ex = p.mapping(z)
    μ = ex[1:p.xlength,:]
    σ = ex[p.xlength+1:end,:]
    return μ, σ .* σ
end

function mean_var(p::CGaussian{T,UnitVar}, z::AbstractArray) where T
    μ = p.mapping(z)
    #σ2 = SVector{xlength(p)}(fill!(similar(μ, xlength(p)), 1)) TODO: use StaticArray
    σ2 = fill!(similar(μ, xlength(p)), 1)
    return μ, σ2
end

function Base.show(io::IO, p::CGaussian{T,V}) where T where V
    e = repr(p.mapping)
    e = sizeof(e)>50 ? "($(e[1:47])...)" : e
    msg = "CGaussian{$T,$V}(xlength=$(p.xlength), zlength=$(p.zlength), mapping=$e)"
    print(io, msg)
end
