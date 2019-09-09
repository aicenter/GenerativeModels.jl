export VAE, XVar, DiagVar, ScalarVar, UnitVar
export prior_mean, prior_variance, prior_mean_var
export elbo

abstract type XVar end
abstract type DiagVar <: XVar end
abstract type ScalarVar <: XVar end
abstract type UnitVar <: XVar end

struct VAE{T,V<:XVar} <: AbstractVAE{T}
    xsize::Int
    zsize::Int
    encoder
    decoder
end

Flux.@treelike VAE

"""
    VAE{T,XVar}(xsize, zsize, encoder, decoder)

Vanilla Variational Auto-Encoder. 

# Arguments
- `T:=Float32`
- `XVar:=UnitVar`: decoder variance variant, one of [UnitVar, ScalarVar, DiagVar]
- `xsize::Int`: input size
- `zsize::Int`: latent size
- `encoder`: an encoder structure (e.g. a Flux Chain)
- `decoder`: decoder structure
"""
VAE{T}(xsize::Int, zsize::Int, encoder, decoder) where T = VAE{T,UnitVar}(xsize, zsize, encoder, decoder)
VAE(xsize::Int, zsize::Int, encoder, decoder)= VAE{Float32,UnitVar}(xsize, zsize, encoder, decoder)
# add a different constructor where XVar is defined as an additional argument?

prior_mean(m::VAE{T}) where T = zeros(T, m.zsize)
prior_variance(m::VAE) = I
prior_mean_var(m::VAE) = (prior_mean(m), prior_variance(m))

function Base.show(io::IO, m::VAE{T,V}) where T where V
    e = summary(m.encoder)
    e = sizeof(e)>80 ? e[1:77]*"..." : e
    d = summary(m.decoder)
    d = sizeof(d)>80 ? d[1:77]*"..." : d
    msg = """VAE{$T,$V}:
      xsize   = $(m.xsize)
      zsize   = $(m.zsize)
      encoder = $(e)
      decoder = $(d)
    """
    print(io, msg)
end

# encoder methods
"""
    encoder_mean_var(m::VAE{T,V}, x::AbstractArray)

Returns the encoder mean and variance as the first and second half of the last encoder layer.
"""
function encoder_mean_var(m::VAE{T}, x::AbstractArray) where T
    ex = m.encoder(x)
    return ex[1:m.zsize,:],  softplus_safe.(ex[m.zsize+1:end,:],T)
end
# probably implement the sample and loglikelihood here... or in the distributions?


# decoder methods
"""
    decoder_mean_var(m::VAE{T,V}, z::AbstractArray) where T

Returns the encoder mean and variance based on V.    
"""
decoder_mean_var(m::VAE{T,UnitVar}, z::AbstractArray) where T = m.decoder(z), I
function decoder_mean_var(m::VAE{T,ScalarVar}, z::AbstractArray) where T
    dz = m.decoder(z)
    return dz[1:m.xsize,:], softplus_safe.(dz[m.xsize+1:end,:],T) # should this return 1D or 2D array?
end
# now these two methods are the same, but they could theoretically be defined in a different way
function decoder_mean_var(m::VAE{T,DiagVar}, z::AbstractArray) where T
    dz = m.decoder(z)
    return dz[1:m.xsize,:], softplus_safe.(dz[m.xsize+1:end,:],T)
end

"""`elbo(m::VAE, x::AbstractArray)`

Computes variational lower bound.
"""
function elbo(m::VAE{T}, x::AbstractArray, β=1.0)
    N  = size(x, 2)
    (μz, σ2z) = encoder_mean_var(m, x)
    z = encoder_sample(m, μz, σ2z)

    llh = -mean(decoder_loglikelihood(m, x, z))
    KL = mean(μz.^2 .+ σ2z - log.(σ2z))/2

    loss = llh + T(β)*KL
end

#function mvhistory_callback(h::MVHistory, m::VAE, lossf::Function, test_data::AbstractArray)
#    function callback()
#        (μz, σz) = encoder_mean_var(m, test_data)
#        σe = m.σe[1]
#        xrec = decoder_mean(m, μz)
#        loss = lossf(test_data)
#        ntuple = DrWatson.@ntuple μz σz xrec loss σe
#        GenerativeModels.push_ntuple!(h, ntuple)
#    end
#end
