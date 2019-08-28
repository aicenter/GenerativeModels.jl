export ARDVAE
export encoder_sample, decoder_sample
export prior_mean, prior_variance, prior_mean_var, prior_sample
export elbo

struct ARDVAE{T<:Real} <: AbstractVAE
    xsize::Int
    zsize::Int
    encoder  # posterior mean
    decoder
    λz::Tracker.TrackedArray{T,1} # prior variance
    σz::Tracker.TrackedArray{T,1} # posterior variance
    σe::Tracker.TrackedArray{T,1}
end

Flux.@treelike ARDVAE

"""`ARDVAE(xsize, zsize, encoder, decoder)`

AutoEncoder that enforces sparsity on the latent layer.
p(x|z) = N(x|z, σe);
p(z)   = N(z|0, λz);
σz: point estimate
"""
function ARDVAE{T}(xsize::Int, zsize::Int, encoder, decoder) where T
    σz = param(ones(T, zsize))
    λz = param(ones(T, zsize))
    σe = param(ones(T, 1))
    ARDVAE(xsize, zsize, encoder, decoder, λz, σz, σe)
end


prior_variance(m::ARDVAE) = softplus.(m.λz)
prior_mean_var(m::ARDVAE) = (UniformScaling(0), softplus.(m.λz))
prior_sample(m::ARDVAE{T}) where T = randn(T, m.zsize) .* m.λz
prior_loglikelihood(m::ARDVAE) = error("Not implemented.")


"""`elbo(m::ARDVAE, x::AbstractArray)`

Computes variational lower bound.
"""
function elbo(m::ARDVAE{T}, x::AbstractArray) where T
    N  = size(x, 2)
    σe = m.σe[1] # TODO: what if σe is not scalar?
    λz = m.λz
    (μz, σz) = encoder_mean_var(m, x)
    # z = encoder_sample(m, x) TODO: this would recompute μz, σz ... change interface of encoder_sample?
    z = μz .+ σz .* randn(T, m.zsize)

    llh = decoder_loglikelihood(m, x, z) / N
    KLz = (sum(2 .* log.(λz ./ σz)) + sum((σz ./ λz).^2) + sum((μz ./ λz).^2)) / N

    loss = llh + KLz + log(σe)*m.zsize/2
end


function Base.show(io::IO, m::ARDVAE{T}) where T
    e = summary(m.encoder)
    e = sizeof(e)>80 ? e[1:77]*"..." : e
    d = summary(m.decoder)
    d = sizeof(d)>80 ? d[1:77]*"..." : d
    msg = """ARDARDVAE{$T}:
      xsize   = $(m.xsize)
      zsize   = $(m.zsize)
      encoder = $(e)
      decoder = $(d)
      σz      = $(m.σz)
      σe      = $(m.σe)
    """
    print(io, msg)
end

function mvhistory_callback(h::MVHistory, m::ARDVAE, lossf::Function, test_data::AbstractArray)
    function callback()
        (μz, σz) = encoder_mean_var(m, test_data)
        λz = m.λz
        σe = m.σe[1]
        xrec = decoder_mean(m, μz)
        loss = lossf(test_data)
        ntuple = DrWatson.@ntuple μz σz λz xrec loss σe
        GenerativeModels.push_ntuple!(h, ntuple)
    end
end


