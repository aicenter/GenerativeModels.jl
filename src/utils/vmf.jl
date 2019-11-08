# Utils for Von Mises-Fisher distribution

"""
	vmfentropy(d, κ)

Entropy of a Von Mises-Fisher distribution with dimensinality `d` and concentration `κ`
"""
vmfentropy(d, κ) = .-κ .* besselix(d / 2, κ) ./ besselix(d / 2 - 1, κ) .- ((d ./ 2 .- 1) .* log.(κ) .- (d ./ 2) .* log(2π) .- (κ .+ log.(besselix(d / 2 - 1, κ))))

"""
	huentropy(d)

Entropy of a Hyperspherical Uniform distribution with dimensinality `d`
"""
huentropy(d) = d / 2 * log(π) + log(2) - lgamma(d / 2)

# Likelihood estimation of a sample x under VMF with given parameters taken from https://pdfs.semanticscholar.org/2b5b/724fb175f592c1ff919cc61499adb26996b1.pdf

"""
    vmf_norm_const(d, κ)

Likelihood normalizing constant of a Von Mises-Fisher distribution with dimensinality `d` and concentration `κ`
"""
vmf_norm_const(d, κ) = κ ^ (d / 2 - 1) / ((2π) ^ (d / 2) * besseli(d / 2 - 1, κ))

# log likelihood of one sample under the VMF dist with given parameters
"""
    log_vmf(x, μ, κ)

Loglikelihood of `x` under the Von Mises-Fisher distribution with mean `μ` and concentration `κ`
"""
log_vmf(x, μ, κ) = κ * μ' * x .+ log(vmf_norm_const(length(μ), κ))

#? Will we need these as well? Can we actually make this without the for cycle? They can probably be optimised by computing the norm constant just once etc.
log_vmf(x::AbstractMatrix, μ::AbstractMatrix, κ::T) where {T <: Number} = [log_vmf(x[:, i], μ[:, i], κ) for i in size(x, 2)] 
log_vmf(x::AbstractMatrix, μ::AbstractMatrix, κ::AbstractVector) = [log_vmf(x[:, i], μ[:, i], κ[i]) for i in size(x, 2)] 

"""
    log_vmf_wo_c(x, μ, κ)

Loglikelihood of `x` under the Von Mises-Fisher distribution with mean `μ` and concentration `κ` **without** the normalizing constant.
It can be very useful when it is used just for comparison of likelihoods etc. because it is very expensive to compute and in many applications
it has no effect on the outcome.
"""
log_vmf_wo_c(x, μ, κ) = κ * μ' * x


# This sampling procedure is one that can be differentiated through taken from https://arxiv.org/pdf/1804.00891.pdf
normalizecolumns(m::AbstractArray{T, 2}) where {T} = m ./ sqrt.(sum(m .^ 2, dims = 1) .+ eps(T))

sample_vmf(μ::AbstractArray{T}, κ::Union{T, AbstractArray{T}}) where {T} = sample_vmf(μ, κ, size(μ, 1))
function sample_vmf(μ::AbstractArray{T}, κ::Union{T, AbstractArray{T}}, dims) where {T}
    ω = sampleω(κ, dims)
    v = zeros(T, dims - 1, size(μ, 2))
	randn!(v)
	v = normalizecolumns(v)
	householderrotation(vcat(ω, sqrt.(1 .- ω .^ 2) .* v), μ)
end

function sampleω(κ::Union{T, AbstractArray{T}}, dims) where {T}
	c = @. sqrt(4κ ^ 2 + (dims - 1) ^ 2)
	b = @. (-2κ + c) / (dims - 1)
	a = @. (dims - 1 + 2κ + c) / 4
	d = @. (4 * a * b) / (1 + b) - (dims - 1) * log(dims - 1)
	ω = rejectionsampling(dims, a, b, d, κ)
end

function householderrotation(zprime::AbstractArray{T}, μ::AbstractArray{T}) where {T}
	e1 = similar(μ) .= 0
	e1[1, :] .= 1
	u = e1 .- μ
	normalizedu = normalizecolumns(u)
	zprime .- 2 .* sum(zprime .* normalizedu, dims = 1) .* normalizedu
end

function rejectionsampling(dims, a, b, d, κ::Union{T, AbstractArray{T}}) where {T}
	beta = Beta((dims - 1) / 2, (dims - 1) / 2)
    ϵ = Adapt.adapt(T, rand(beta, size(a)...)) #! This is really stupid but even Beta{Float32} samples Float64 values :( - we should switch to our sampler
    u = rand(T, size(a)...)

	accepted = isaccepted(ϵ, u, dims, a, b, d)
	it = 0
	while (!all(accepted)) & (it < 10000)
		mask = .! accepted
		ϵ[mask] = Adapt.adapt(T, rand(beta, sum(mask))) #! same issue as above
		u[mask] = rand(T, sum(mask))
		accepted[mask] = isaccepted(mask, ϵ, u, dims, a, b, d)
		it += 1
	end
	if it >= 10000
		println("Warning - sampler was stopped by 10000 iterations - it did not accept the sample!")
        # perhaps this can be removed but some networks were causing issues in too high kappas etc so it was better to let 
        # it continue with a bit imprecise number from time to time than crashing the whole computation - might be an issue though
	end
	return @. (1 - (1 + b) * ϵ) / (1 - (1 - b) * ϵ)
end

isaccepted(mask, ϵ, u, dims::Int, a, b, d) = isaccepted(ϵ[mask], u[mask], dims, a[mask], b[mask], d[mask]);
function isaccepted(ϵ, u, dims::Int, a, b, d)
	ω = @. (1 - (1 + b) * ϵ) / (1 - (1 - b) * ϵ)
	t = @. 2 * a * b / (1 - (1 - b) * ϵ)
	@. (dims - 1) * log(t) - t + d >= log(u)
end