export train!, softplus_safe
export diffeq_rd

# for over/underflow in logs
"""
	softplus_safe(x,T=Float32)

A softplus with small additive constant for safe operations.
"""
softplus_safe(x,T=Float32) = softplus(x) .+ T(1e-6)

### Flux chain builders ###
"""
    function layer_builder(d,k,o,n,ftype,lastlayer = "",ltype = "Dense")

Create a chain with `n` layers of with `k` neurons with activation function `ftype`.
Input and output dimension is `d` / `o`.
If lastlayer is not specified, all layers use the same function.
If lastlayer is "linear", then the last layer is forced to be Dense with linear activation.

It is also possible to specify dimensions in a vector.
"""
layer_builder(k::Vector{Int},l::Vector,f::Vector) = Flux.Chain(map(i -> i[1](i[3],i[4],i[2]),zip(l,f,k[1:end-1],k[2:end]))...)

layer_builder(d::Int,k::Int,o::Int,n::Int, args...) =
    layer_builder(vcat(d,fill(k,n-1)...,o), args...)

function layer_builder(ks::Vector{Int},ftype::String,lastlayer::String = "",ltype::String = "Dense")
    ftype = (ftype == "linear") ? "identity" : ftype
    ls = Array{Any}(fill(eval(:($(Symbol(ltype)))),length(ks)-1))
    fs = Array{Any}(fill(eval(:($(Symbol(ftype)))),length(ks)-1))
    if !isempty(lastlayer)
        fs[end] = (lastlayer == "linear") ? identity : eval(:($(Symbol(lastlayer))))
        ls[end] = (lastlayer == "linear") ? Dense : ls[end]
    end
    layer_builder(ks,ls,fs)
end

"""
    stack_layers(lsize, activation, [layer=Dense; last=identity])

Construct a Flux chain (e.g. encoder/decoder) consisting of `length(lsize)-1` layers of 
type `layer` with  `activation` in between. Default last activation is `identity`. 
Width of layers is defined by the vector `lsize`.
"""
stack_layers(lsize::Union{Vector, Tuple}, activation, layer=Dense; last=identity) =  
    layer_builder([x for x in lsize], 
        Array{Any}(fill(layer, size(lsize,1)-1)), 
        Array{Any}([fill(activation, size(lsize,1)-2); last]))

"""
    conv_encoder(xsize, zsize, kernelsizes, channels, scalings[; activation, densedims])

Constructs a convolutional encoder.

# Arguments
- `xsize`: size of input - (h,w,c)
- `zsize`: latent space dimension
- `kernelsizes`: kernelsizes for conv. layers (only odd numbers are allowed)
- `channels`: channel numbers
- `scalings`: scalings vector
- `activation`: default relu
- `densedims`: if set, more than one dense layers are used 

# Example
```julia-repl
julia> encoder = conv_encoder((64, 48, 1), 2, (3, 5, 5), (2, 4, 8), (2, 2, 2), densedims = (256))
Chain(Conv((3, 3), 1=>2, relu), MaxPool((2, 2), pad = (0, 0, 0, 0), stride = (2, 2)), Conv((5, 5), 2=>4, relu), MaxPool((2, 2), pad = (0, 0, 0, 0), stride = (2, 2)), Conv((5, 5), 4=>8, relu), MaxPool((2, 2), pad = (0, 0, 0, 0), stride = (2, 2)), #44, Dense(384, 2))

julia> encoder(randn(Float32, 64, 48, 1, 2))
2×2 Array{Float32,2}:
  0.247844   0.133781
 -0.605763  -0.494911
```
"""
function conv_encoder(xsize::Union{Tuple, Vector}, zsize::Int, kernelsizes::Union{Tuple, Vector}, 
    channels::Union{Tuple, Vector}, scalings::Union{Tuple, Vector}; 
    activation = relu, densedims::Union{Tuple, Vector} = [])
    nconv = length(kernelsizes)
    (nconv == length(channels) == length(scalings)) ? nothing : error("incompatible input dimensions")
    (length(xsize) == 3) ? nothing : error("xsize must be (h, w, c)")
    # also check that kernelsizes are all odd numbers
    (all(kernelsizes .% 2 .== 1)) ? nothing : error("not implemented for even kernelsizes")

    # initialize some stuff
    cins = vcat(xsize[3], channels[1:end-1]...) # channels in
    couts = channels # channels out
    ho = xsize[1]/(reduce(*, scalings)) # height before reshaping
    wo = xsize[2]/(reduce(*, scalings)) # width before reshaping
    (ho == floor(Int, ho)) ? ho = floor(Int, ho) : error("your input size and scaling is not compatible")
    (wo == floor(Int, wo)) ? wo = floor(Int, wo) : error("your input size and scaling is not compatible")
    din = ho*wo*channels[end]

    # now build a vector of layers to be used later
    layers = Array{Any,1}()
    # first add the convolutional and maxpooling layers
    for (k, ci, co, s) in zip(kernelsizes, cins, couts, scalings)
        pad = Int((k-1)/2)
        # paddding so that input and output size are same
        push!(layers, Conv((k,k), ci=>co, activation; pad = (pad, pad))) 
        push!(layers, MaxPool((s,s)))
    end

    # reshape
    push!(layers, x -> reshape(x, din, :))

    # and dense layers
    ndense = length(densedims)
    dins = vcat(din, densedims...)
    douts = vcat(densedims..., zsize)
    dacts = vcat([activation for _ in 1:ndense]..., identity)
    for (_di, _do, _da) in zip(dins, douts, dacts)
        push!(layers, Dense(_di, _do, _da))
    end

    Flux.Chain(layers...)
end

"""
    conv_decoder(xsize, zsize, kernelsizes, channels, scalings[; activation, densedims])

Constructs a convolutional encoder.

# Arguments
- `xsize`: size of input - (h,w,c)
- `zsize`: latent space dimension
- `kernelsizes`: kernelsizes for conv. layers (only odd numbers are allowed)
- `channels`: channel numbers
- `scalings`: scalings vector
- `activation`: default relu
- `densedims`: if set, more than one dense layers are used 

# Example
```julia-repl
julia> decoder = conv_decoder((64, 48, 1), 2, (5, 5, 3), (8, 4, 2), (2, 2, 2); densedims = (256))
Chain(Dense(2, 256, relu), Dense(256, 384, relu), #19, ConvTranspose((2, 2), 8=>8, relu), Conv((5, 5), 8=>4, relu), ConvTranspose((2, 2), 4=>4, relu), Conv((5, 5), 4=>2, relu), ConvTranspose((2, 2), 2=>2, relu), Conv((3, 3), 2=>1))

julia> y = decoder(randn(Float32, 2, 2));

julia> size(y)
(64, 48, 1, 2)
```
"""
function conv_decoder(xsize::Union{Tuple, Vector}, zsize::Int, kernelsizes::Union{Tuple, Vector}, 
    channels::Union{Tuple, Vector}, scalings::Union{Tuple, Vector}; 
    activation = relu, densedims::Union{Tuple, Vector} = [])
    nconv = length(kernelsizes)
    (nconv == length(channels) == length(scalings)) ? nothing : error("incompatible input dimensions")
    (length(xsize) == 3) ? nothing : error("xsize must be (h, w, c)")
    # also check that kernelsizes are all odd numbers
    (all(kernelsizes .% 2 .== 1)) ? nothing : error("not implemented for even kernelsizes")

    # initialize some stuff
    cins = channels # channels in
    couts = vcat(channels[2:end]..., xsize[3]) # channels out
    ho = xsize[1]/(reduce(*, scalings)) # height after reshaping
    wo = xsize[2]/(reduce(*, scalings)) # width after reshaping
    (ho == floor(Int, ho)) ? ho = floor(Int, ho) : error("your input size and scaling is not compatible")
    (wo == floor(Int, wo)) ? wo = floor(Int, wo) : error("your input size and scaling is not compatible")
    dout = ho*wo*channels[1]

    # now build a vector of layers to be used later
    layers = Array{Any,1}()

    # start with dense layers
    ndense = length(densedims)
    dins = vcat(zsize, densedims...)
    douts = vcat(densedims..., dout)
    for (_di, _do) in zip(dins, douts)
        push!(layers, Dense(_di, _do, activation))
    end

    # reshape
    push!(layers, x -> reshape(x, ho, wo, channels[1], :))

    # add the transpose nad convolutional layers
    acts = vcat([activation for _ in 1:nconv-1]..., identity) 
    for (k, ci, co, s, act) in zip(kernelsizes, cins, couts, scalings, acts)
        pad = Int((k-1)/2)
        # use convtranspose for upscaling - there are other posibilities, however this seems to be ok
        push!(layers, ConvTranspose((s,s), ci=>ci, activation, stride = (s,s))) 
        push!(layers, Conv((k,k), ci=>co, act; pad = (pad, pad)))
    end

    Flux.Chain(layers...)
end

### training ###
"""
    update_params!(model, data, loss, opt)

Basic training step - computation of the loss, backpropagation of gradients and optimisation 
of weights. The loss and opt arguments can be arrays/lists/tuples.
"""
function update_params!(model, data, loss, opt)
    ps = Flux.params(model)
    gs = gradient(ps) do
        loss(data...)
    end
    Flux.Optimise.update!(opt, ps, gs)
end 

"""
    train!(model, data, loss, optimiser, callback; [usegpu, memory_efficient])

Train the model. Function callback(model, batch, loss, opt) is 
called every iteration - use it to store or print training progress, stop training etc. 
"""
function train!(model, data, loss, optimiser, callback; 
    usegpu = false, memory_efficient = false)
    for _data in data
        try
            if usegpu
             _data = _data |> gpu
            end
            update_params!(model, _data, loss, optimiser)
            # now call the callback function
            # can be an object so it can store some values between individual calls
            callback(model, _data, loss, optimiser)
        catch e
            # setup a special kind of exception for known cases with a break
            rethrow(e)
        end
        if memory_efficient
            # this might be important for conv nets running on gpu
            # large nets might still train but slowly
            GC.gc();
        end
    end
end

### GAN ###
"""
    gen_loss([T=Float32], sg)

Generator loss for generated data score `sg`.
"""
generator_loss(T,sg) = - mean(log.(sg) .+ eps(T))
generator_loss(sg) = generator_loss(Float32,sg)

"""
    disc_loss([T=Float32], st, sg)

Discriminator loss for true data score `st` and generated data score `sg`.
"""
discriminator_loss(T,st,sg) = - T(0.5)*(mean(log.(st .+ eps(T))) + mean(log.(1 .- sg) .+ eps(T)))
discriminator_loss(st,sg) = discriminator_loss(Float32,st,sg)


"""
    destructure(m)

Returns all parameters of a Flux model in one long vector.
This includes **all** `AbstractArray` fields
(Adapted from DiffEqFlux.jl)
"""
function destructure(m)
    xs = []
    Flux.fmap(m) do x
        x isa AbstractArray && push!(xs, x)
        return x
    end
    return vcat(vec.(xs)...)
end

"""
    restructure(m, xs::AbstractVector)

Populate a Flux model with parameters as given in a long vector of xs.
xs must include **all** `AbstractArray` fields.
(Adapted from DiffEqFlux.jl)
"""
function restructure(m, xs::AbstractVector)
    i = 0
    Flux.fmap(m) do x
        x isa AbstractArray || return x
        x = reshape(xs[i.+(1:length(x))], size(x))
        i += length(x)
        return x
    end
end
