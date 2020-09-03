export MMD, mmd
struct MMD{K<:Kernel} <: PreMetric
    kernel::K
end

function (m::MMD)(x::AbstractArray, y::AbstractArray)
    xx = sum(kernelmatrix(m.kernel, x))
    yy = sum(kernelmatrix(m.kernel, y))
    xy = sum(kernelmatrix(m.kernel, x, y))
    xx + yy - 2xy
end


"""
	mmd(Kernel(γ), x, y)
	mmd(Kernel(γ), x, y, n)

MMD with Gaussian kernel of bandwidth `γ` using at most `n` samples
"""
mmd(k::Kernel, x::AbstractArray, y::AbstractArray) = MMD(k)(x, y)
mmd(k::Kernel, x::AbstractArray, y::AbstractArray, n::Int) =
    mmd(k, samplecolumns(x,n), samplecolumns(y,n))


struct IMQKernel <: KernelFunctions.SimpleKernel end
KernelFunctions.kappa(::IMQKernel, d2::Real) = 1 / (1+d2)
KernelFunctions.metric(::IMQKernel) = SqEuclidean()
