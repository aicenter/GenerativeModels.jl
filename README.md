[![Build Status](https://travis-ci.com/aicenter/GenerativeModels.jl.svg?branch=master)](https://travis-ci.com/aicenter/GenerativeModels.jl)
[![codecov](https://codecov.io/gh/aicenter/GenerativeModels.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/aicenter/GenerativeModels.jl)

# GenerativeModels.jl

This library contains a collection of generative models for anomaly detection.
It defines learnable (conditional) distributions that can be used in conjuction
with [`Flux.jl`](https://github.com/FluxML/Flux.jl) and
[`Zygote.jl`](https://github.com/FluxML/Zygote.jl), that aims to make
experimenting with new models easy.

As an example, check out how to build a conventional variational autoencoder
with a diagonal variance on the latent dimension and a scalar variance on the
reconstruction:

```julia
using GenerativeModels
using Flux

xlen  = 5
zlen  = 2
dtype = Float32

μ = NoGradArray(zeros(dtype, zlen))  # NoGradArray is filtered when calling `Flux.params`
σ = NoGradArray(ones(dtype, zlen))
prior = Gaussian(μ, σ)

encoder = Dense(xlen, zlen*2)  # encoder returns mean and diagonal variance
encoder_dist = CMeanVarGaussian{DiagVar}(encoder)

decoder = Dense(zlen, xlen+1)  # decoder returns mean and scalar variance
decoder_dist = CMeanVarGaussian{ScalarVar}(decoder)

vae = VAE(prior, encoder_dist, decoder_dist)
length(params(vae)) == 4  # we only get trainable params from the two Dense layers
```

Now you have a model that you can call `params(vae)` on and use Flux as you are
used to. You can also easily sample from it once you are done training:

```julia
z = rand(vae.prior, 10)   # sample 10 times from the prior
μ = mean(vae.decoder, z)  # get decoder means
x = rand(vae.decoder, z)  # get decoder samples
```

But say, you want to learn the variance of your prior during training... Easy!
Just turn the prior variance into a normal `Array`:
```julia
trainable_prior = Gaussian(NoGradArray(zeros(zlen)), ones(zlen))

vae = VAE(trainable_prior, encoder_dist, decoder_dist)
length(params(vae)) == 5  # prior variance is now included in trainable params
```

Done!


## Development

To start julia with the exact package versions that are specified in the
dependencies run `julia --project` from the root of this repo.

Where possible, custom checkpointing/other convenience functions should be using
[`DrWatson.jl`](https://juliadynamics.github.io/DrWatson.jl/stable/)
functionality such as `tagsave` to ensure reproducability of simulations.


### Structure

    |-src
    |  |- models
    |  |- pdfs
    |  |- anomaly_scores
    |  |- utils
    |-test

The models themselves are defined in `src/models`. Each file contains a
specific model that inherits from `AbstractGM` and has three fields:
```julia
struct Model{T} <: AbstractGM
    prior::AbstractPDF
    encoder::AbstractCPDF
    decoder::AbstractCPDF
end
```

and implements e.g. custom loss functions.


### Model / distribution interface

The distributions used for prior, encoder, and decoder all implement a common
interface that includes the functions `mean`, `variance`, `mean_var`, `rand`,
`loglikelihood`, `kld`.
This interface makes it possible that functions such as the ELBO or the anomaly
scores can be generalized. E.g. the ELBO code looks like this:

```julia
function elbo(m::AbstractVAE, x::AbstractArray; β=1)
    z = rand(m.encoder, x)
    llh = mean(-loglikelihood(m.decoder, x, z))
    kl  = mean(kld(m.encoder, m.prior, x))
    llh + β*kl
end
```

Check out `src/pdf/abstract_pdfs.jl` for the fully defined interface.
