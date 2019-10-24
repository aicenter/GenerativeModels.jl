export CMeanVarGaussian
export mean_var

"""
    CMeanVarGaussian{T,AbstractVar}(mapping)

Conditional Gaussian that maps an input z to a mean μx and a variance σ2x.
The mapping must output dimensions appropriate for the chosen variance type:
- DiagVar: μx = mapping(z)[1:end/2]; σ2 = mapping(z)[end/2+1:end]
- ScalarVar: μx = mapping(z)[1:end-1]; σ2 = mapping(z)[end:end]

# Arguments
- `mapping`: maps condition z to mean and variance (e.g. a Flux Chain)
- `AbstractVar`: one of the variance types: DiagVar, ScalarVar
- `T`: expected eltype. E.g. `rand` will try to sample arrays of this eltype.
  If the mapping returns a different eltype the output of `mean`,`variance`,
  and `rand` is not necessarily of eltype T.

# Example
```julia-repl
julia> p = CMeanVarGaussian{Float32,ScalarVar}(Dense(2, 3))
CMeanVarGaussian{Float32,ScalarVar}(mapping=Dense(2, 3))

julia> mean_var(p, ones(2))
(Float32[1.6191938; -0.437356], Float32[4.131034])

julia> rand(p, ones(2))
2×1 Array{Float32,2}:
 0.7168678
 0.16322285
```
"""
struct CMeanVarGaussian{T,V<:AbstractVar} <: AbstractCGaussian{T}
    mapping
end

function mean_var(p::CMeanVarGaussian{T,DiagVar}, z::AbstractArray) where T
    ex = p.mapping(z)
    if eltype(ex) != T
        error("Mapping should return eltype $T. Found: $(eltype(ex))")
    end

    xlen = Int(size(ex, 1) / 2)
    μ = ex[1:xlen,:]
    σ = ex[xlen+1:end,:]

    return μ, σ .* σ
end

function mean_var(p::CMeanVarGaussian{T,ScalarVar}, z::AbstractArray) where T
    ex = p.mapping(z)
    if eltype(ex) != T
        error("Mapping should return eltype $T. Found: $(eltype(ex))")
    end

    μ = ex[1:end-1,:]
    σ = ex[end:end,:] .* fill!(similar(μ, 1, size(μ,2)), 1)

    return μ, σ .* σ
end

# make sure that parameteric constructor is called...
function Flux.functor(p::CMeanVarGaussian{T,V}) where {T,V}
    fs = fieldnames(typeof(p))
    nt = (; (name=>getfield(p, name) for name in fs)...)
    nt, y -> CMeanVarGaussian{T,V}(y...)
end

function Base.show(io::IO, p::CMeanVarGaussian{T,V}) where T where V
    e = repr(p.mapping)
    e = sizeof(e)>50 ? "($(e[1:47])...)" : e
    msg = "CMeanVarGaussian{$T,$V}(mapping=$e)"
    print(io, msg)
end
