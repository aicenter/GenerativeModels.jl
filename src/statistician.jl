"""
    NeuralStatistician(IE, [clength::Int, pc], enc_c, cond_z, enc_z, dec)

Neural Statistician model from https://arxiv.org/abs/1606.02185
paper Towards a Neural Statistician.

Acts on bag data (sets of instances).

# Arguments
* `IE`: instance encoder (trainable neural network)
* `pc`: MvNormal prior p(c)
* `clength`: dimension of context prior MvNormal distribution
* `enc_c`: context encoder q(c|D)
* `cond_z`: conditional p(z|c)
* `enc_z`: instance encoder q(z|x,c)
* `dec`: decoder p(x|z)

# Example
Create a Neural Statistician model:
```julia-repl
julia> idim, vdim, cdim, zdim = 5, 3, 2, 4
julia> instance_enc = Chain(Dense(idim,15,swish),Dense(15,vdim))
julia> enc_c = SplitLayer(vdim,[cdim,1])
julia> enc_c_dist = ConditionalMvNormal(enc_c)
julia> cond_z = SplitLayer(cdim,[zdim,1])
julia> cond_z_dist = ConditionalMvNormal(cond_z)
julia> enc_z = SplitLayer(cdim+vdim,[zdim,1])
julia> enc_z_dist = ConditionalMvNormal(enc_z)
julia> dec = SplitLayer(zdim,[idim,1])
julia> dec_dist = ConditionalMvNormal(dec)

julia> model = NeuralStatistician(instance_enc, cdim, enc_c_dist, cond_z_dist, enc_z_dist, dec_dist)

julia> bag = randn(idim,12)
julia> loss(x) = -elbo(model,x)
julia> loss(bag)
10430.707315113537
```
"""
struct NeuralStatistician{IE,pc <: ContinuousMultivariateDistribution,qc <: ConditionalMvNormal,pz <: ConditionalMvNormal,qz <: ConditionalMvNormal,D <: ConditionalMvNormal} # <: AbstractNS
    instance_encoder::IE
    prior_c::pc
    encoder_c::qc
    conditional_z::pz
    encoder_z::qz
    decoder::D
end

Flux.@functor NeuralStatistician

function Flux.trainable(m::NeuralStatistician)
    (instance_encoder = m.instance_encoder, encoder_c = m.encoder_c, conditional_z = m.conditional_z, encoder_z = m.encoder_z, decoder = m.decoder)
end

function NeuralStatistician(IE, clength::Int, enc_c::ConditionalMvNormal, cond_z::ConditionalMvNormal, enc_z::ConditionalMvNormal, dec::ConditionalMvNormal)
    W = first(Flux.params(enc_c))
    μ_c = fill!(similar(W, clength), 0)
    σ_c = fill!(similar(W, clength), 1)
    prior_c = DistributionsAD.TuringMvNormal(μ_c, σ_c)
    NeuralStatistician(IE, prior_c, enc_c, cond_z, enc_z, dec)
end


"""
    elbo(m::NeuralStatistician,x::AbstractArray; β1=1.0, β2=1.0)

Neural Statistician log-likelihood lower bound.

For a Neural Statistician model, simply create a loss
function as
    
    `loss(x) = -elbo(model,x)`

where `model` is a NeuralStatistician type.

The β terms scale the KLDs:
* β1: KL[q(c|D) || p(c)]
* β2: KL[q(z|c,x) || p(z|c)]
"""
function elbo(m::NeuralStatistician, x::AbstractArray;β1=1.0,β2=1.0)
    # instance network
    v = m.instance_encoder(x)
    p = mean(v, dims=2)

    # sample latent for context
    c = rand(m.encoder_c, p)

    # sample latent for instances
    h = hcat([vcat(v[1:end,i], c) for i in 1:size(v, 2)]...)
    z = rand(m.encoder_z, h)
	
    # 3 terms - likelihood, kl1, kl2
    llh = mean(logpdf(m.decoder, x, z))
    kld1 = mean(kl_divergence(condition(m.encoder_c, v), m.prior_c))
    kld2 = mean(kl_divergence(condition(m.encoder_z, h), condition(m.conditional_z, c)))
    llh - β1 * kld1 - β2 * kld2
end

function Base.show(io::IO, m::NeuralStatistician)
    IE = repr(m.instance_encoder)
    IE = sizeof(IE) > 70 ? "($(IE[1:70 - 3])...)" : IE
    pc = repr(m.prior_c)
    pc = sizeof(pc) > 70 ? "($(pc[1:70 - 3])...)" : pc
    E_c = repr(m.encoder_c)
    E_c = sizeof(E_c) > 70 ? "($(E_c[1:70 - 3])...)" : E_c
    cond_z = repr(m.conditional_z)
    cond_z = sizeof(cond_z) > 70 ? "($(cond_z[1:70 - 3])...)" : E_z
    E_z = repr(m.encoder_z)
    E_z = sizeof(E_z) > 70 ? "($(E_z[1:70 - 3])...)" : E_z
    D = repr(m.decoder)
    D = sizeof(D) > 70 ? "($(D[1:70 - 3])...)" : D

    msg = """$(nameof(typeof(m))):
     instance_encoder = $(IE)
     prior_c          = $(pc)
     encoder_c        = $(E_c)
     conditional_z    = $(cond_z)
     encoder_z        = $(E_z)
     decoder          = $(D)
    """
    print(io, msg)
end

# needs to extend kl_divergence from IPMeasures
# (m::KLDivergence)(p::ConditionalDists.BMN, q::ConditionalDists.BMN) = IPMeasures._kld_gaussian(p,q)