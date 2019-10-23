export NoGradArray

struct NoGradArray{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
end

Base.size(A::NoGradArray) = size(A.data)
Base.IndexStyle(::Type{<:NoGradArray}) = IndexLinear()
Base.getindex(A::NoGradArray, i::Int) = A.data[i]

function _trainable(m)
    ps = Flux.functor(m)[1]
    (; [k=>ps[k] for k in keys(ps) if !isa(ps[k], NoGradArray)]...)
end

# overload to ignore NoGradArrays
Flux.trainable(m::AbstractCPDF) = _trainable(m)
Flux.trainable(m::AbstractPDF) = _trainable(m)
Flux.trainable(m::AbstractGM) = _trainable(m)
