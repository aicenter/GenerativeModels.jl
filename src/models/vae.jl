export VAE
export prior_mean, prior_variance, prior_mean_var, prior_sample
export prior_loglikelihood
export elbo

struct VAE{T} <: AbstractVAE{T}
    xsize::Int
    zsize::Int
    encoder
    decoder
    σz::Tracker.TrackedArray{T,1}
    σe::Tracker.TrackedArray{T,1}
end

Flux.@treelike VAE

"""`VAE(xsize, zsize, encoder, decoder)`

Variational Auto-Encoder with scalar Gaussian noise on reconstruction.
p(x|z) = N(x|z, σe);
p(z)   = N(z|0, 1);
q(z|x) = N(z|μz, σz)
"""
function VAE{T}(xsize::Int, zsize::Int, encoder, decoder) where T
    σz = param(ones(T, zsize) / 100)
    σe = param(ones(T, 1) / 10)
    VAE(xsize, zsize, encoder, decoder, σz, σe)
end


prior_mean(m::VAE{T}) where T = zeros(T, m.zsize)
prior_variance(m::VAE) = I
prior_mean_var(m::VAE) = (prior_mean(m), prior_variance(m))
prior_sample(m::VAE{T}) where T = randn(T, m.zsize)

function prior_loglikelihood(m::VAE, z::AbstractArray)
    @assert size(z, 1) == m.zsize
    -dropdims(sum(z.^2, dims=1), dims=1) / 2
end


"""`elbo(m::VAE, x::AbstractArray)`

Computes variational lower bound.
"""
function elbo(m::VAE, x::AbstractArray)
    N  = size(x, 2)
    (μz, σz) = encoder_mean_var(m, x)
    z = encoder_sample(m, μz, σz)

    llh = -sum(decoder_loglikelihood(m, x, z)) / N
    KLz = (sum(μz.^2 .+ σz) / 2 - sum(log.(σz))) / N

    σe = decoder_variance(m, z)[1] # TODO: what if σe is not scalar?
    loss = llh + KLz + log(σe)*m.zsize/2
end


function Base.show(io::IO, m::VAE{T}) where T
    e = summary(m.encoder)
    e = sizeof(e)>80 ? e[1:77]*"..." : e
    d = summary(m.decoder)
    d = sizeof(d)>80 ? d[1:77]*"..." : d
    msg = """VAE{$T}:
      xsize   = $(m.xsize)
      zsize   = $(m.zsize)
      encoder = $(e)
      decoder = $(d)
      σz      = $(m.σz)
      σe      = $(m.σe)
    """
    print(io, msg)
end


function mvhistory_callback(h::MVHistory, m::VAE, lossf::Function, test_data::AbstractArray)
    function callback()
        (μz, σz) = encoder_mean_var(m, test_data)
        σe = m.σe[1]
        xrec = decoder_mean(m, μz)
        loss = lossf(test_data)
        ntuple = DrWatson.@ntuple μz σz xrec loss σe
        GenerativeModels.push_ntuple!(h, ntuple)
    end
end
