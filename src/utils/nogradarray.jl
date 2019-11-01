export NoGradArray, NoGradCuArray

"""
    NoGradArray(A::Array)

Wraps an array. Only used to filter out arrays that should stay constant
during training.
"""
struct NoGradArray{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
end

Base.size(A::NoGradArray) = size(A.data)
Base.IndexStyle(::Type{<:NoGradArray}) = IndexLinear()
Base.getindex(A::NoGradArray, idx) = getindex(A.data, idx)



"""
    NoGradCuArray(A::Array)

Wraps a CuArray. Only used to filter out arrays that should stay constant
during training on the GPU.
"""
struct NoGradCuArray{T,N} <: AbstractArray{T,N}
    data::CuArray{T,N}
end

NoGradCuArray(A::Array) = NoGradCuArray(cu(A))

Base.size(A::NoGradCuArray) = size(A.data)
Base.IndexStyle(::Type{<:NoGradCuArray}) = IndexStyle(CuArray)
Base.getindex(A::NoGradCuArray, idx) = getindex(A.data, idx)

Base.similar(a::NoGradCuArray{T,N}) where {T,N} =
    CuArray{T,N}(undef, size(a))
Base.similar(a::NoGradCuArray{T}, dims::Base.Dims{N}) where {T,N} =
    CuArray{T,N}(undef, dims)
Base.similar(a::NoGradCuArray, ::Type{T}, dims::Base.Dims{N}) where {T,N} =
    CuArray{T,N}(undef, dims)


import Base.Broadcast: Broadcasted, BroadcastStyle, ArrayStyle

BroadcastStyle(::Type{<:NoGradCuArray}) = ArrayStyle{NoGradCuArray}()
BroadcastStyle(::ArrayStyle{NoGradCuArray}, ::ArrayStyle{CuArray}) = ArrayStyle{NoGradCuArray}()

function Base.similar(bc::Broadcasted{ArrayStyle{NoGradCuArray}}, ::Type{T}) where T
    similar(CuArray{T}, axes(bc))
end

function Base.similar(bc::Broadcasted{ArrayStyle{NoGradCuArray}}, ::Type{T}, dims...) where {T}
    similar(CuArray{T}, dims...)
end

function Broadcast.broadcasted(::ArrayStyle{NoGradCuArray}, f, args...)
    _args = map(cu, args)
    Broadcasted{ArrayStyle{NoGradCuArray}}(CuArrays.cufunc(f), _args, nothing)
end


# conversions to make Flux.gpu/Flux.cpu work
CuArrays.cu(x::NoGradArray) = NoGradCuArray(x.data)
Core.Array(x::NoGradCuArray) = NoGradArray(Array(x.data))


function _trainable(m)
    ps = Flux.functor(m)[1]
    (; [k=>ps[k] for k in keys(ps) if !(isa(ps[k], NoGradArray) | isa(ps[k], NoGradCuArray))]...)
end

# overload to ignore NoGradArrays
Flux.trainable(m::AbstractCPDF) = _trainable(m)
Flux.trainable(m::AbstractPDF) = _trainable(m)
Flux.trainable(m::AbstractGM) = _trainable(m)
