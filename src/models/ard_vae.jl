export ARDVAE
export prior_mean, prior_variance, prior_mean_var, prior_sample
export prior_loglikelihood
export elbo

struct ARDVAE{T} <: AbstractVAE{T}
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
λz: point estimate;
q(z|x) = N(z|μz, σz)
"""
function ARDVAE{T}(xsize::Int, zsize::Int, encoder, decoder) where T
    σz = param(ones(T, zsize) / 100)
    λz = param(ones(T, zsize))
    σe = param(ones(T, 1) / 10)
    ARDVAE(xsize, zsize, encoder, decoder, λz, σz, σe)
end

prior_mean_var(m::ARDVAE{T}) where T = (zeros(T, m.zsize), m.λz.^2)
encoder_mean_var(m::ARDVAE, x::AbstractArray) = (m.encoder(x), m.σz.^2)
decoder_mean_var(m::ARDVAE, z::AbstractArray) = (m.decoder(z), m.σe.^2)

function prior_loglikelihood(m::ARDVAE, z::AbstractArray)
    @assert size(z, 1) == m.zsize
    σz = prior_variance(m)
    -dropdims(sum(z.^2 ./ σz, dims=1), dims=1) / 2 .+ sum(log.(σz))
end


"""`elbo(m::ARDVAE, x::AbstractArray)`

Computes variational lower bound.
"""
function elbo(m::ARDVAE, x::AbstractArray)
    N = size(x, 2)
    λz = prior_variance(m)
    (μz, σz) = encoder_mean_var(m, x)
    z = encoder_sample(m, μz, σz)

    llh = -sum(decoder_loglikelihood(m, x, z)) / N
    KLz = (sum(log.(λz ./ σz)) + sum(σz ./ λz) + sum(μz.^2 ./ λz)) / N

    σe = decoder_variance(m, z)[1] # TODO: what if σe is not scalar?
    loss = llh + KLz + log(σe)*m.zsize
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
      λz      = $(m.λz)
      σz      = $(m.σz)
      σe      = $(m.σe)
    """
    print(io, msg)
end


function mvhistory_callback(h::MVHistory, m::ARDVAE, lossf::Function, test_data::AbstractArray)
    function callback()
        (μz, σz) = encoder_mean_var(m, test_data)
        λz = m.λz
        σe = decoder_variance(m, μz)
        xrec = decoder_mean(m, μz)
        loss = lossf(test_data)
        ntuple = DrWatson.@ntuple μz σz λz xrec loss σe
        GenerativeModels.push_ntuple!(h, ntuple)
    end
end
