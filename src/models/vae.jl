export VAE
export encoder_sample, decoder_sample
export prior_mean, prior_variance, prior_mean_var, prior_sample
export elbo

struct VAE{T<:Real} <: AbstractVAE
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
"""
function VAE{T}(xsize::Int, zsize::Int, encoder, decoder) where T
    σz = param(ones(T, zsize))
    σe = param(ones(T, 1))
    VAE(xsize, zsize, encoder, decoder, σz, σe)
end

function encoder_sample(m::VAE{T}, x::AbstractArray) where T
    # defined specifically for VAE type because of datatype T...
    (μz, σz) = encoder_mean_var(m, x)
    μz .+ σz .* randn(T, m.zsize)
end


prior_mean(m::VAE) = UniformScaling(0)
prior_variance(m::VAE) = I
prior_mean_var(m::VAE) = (UniformScaling(0), I)
prior_sample(m::VAE{T}) where T = randn(T, m.zsize)

function decoder_sample(m::VAE{T}, z::AbstractArray) where T
    # defined specifically for VAE type because of datatype T...
    (μx, σe) = decoder_mean_var(m, z)
    μx .+ σe .* randn(T, m.xsize)
end


"""`elbo(m::VAE, x::AbstractArray)`

Computes variational lower bound.
"""
function elbo(m::VAE{T}, x::AbstractArray) where T
    N  = size(x, 2)
    σe = m.σe[1] # TODO: what if σe is not scalar?

    (μz, σz) = encoder_mean_var(m, x)
    # z = encoder_sample(m, x) TODO: this would recompute μz, σz ... change interface of encoder_sample?
    z = μz .+ σz .* randn(T, m.zsize)

    llh = decoder_loglikelihood(m, x, z) / N
    KLz = (sum(μz.^2 .+ σz.^2) / 2. - sum(log.(abs.(σz)))) / N

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
