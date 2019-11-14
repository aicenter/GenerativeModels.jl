export HypersphericalUniform

"""
HypersphericalUniform{T}

Hyperspherical uniform distribution in `dims` dimensions.

"""
struct HypersphericalUniform{T} <: AbstractPDF{T}
    dims::Int
end

HypersphericalUniform(d::Int) = HypersphericalUniform{Float32}(d)

length(p::HypersphericalUniform) = p.dims

function rand(p::HypersphericalUniform{T}, batchsize::Int=1) where {T}
    v = randn(T, p.dims, batchsize)
    normalizecolumns(v)
end
