[![Build Status](https://travis-ci.com/nmheim/GenerativeModels.jl.svg?branch=master)](https://travis-ci.com/nmheim/GenerativeModels.jl)

# GenerativeModels.jl

This library contains a collection of generative models for anomaly detection.
To start julia with the exact package versions that are specified in the
dependencies run `julia --project` from the root of this repo.

Where possible, custom checkpointing/other convenience functions should be using
[`DrWatson.jl`](https://juliadynamics.github.io/DrWatson.jl/stable/)
functionality such as `tagsave` to ensure reproducability of simulations.


## Structure

    |-src
    |  |- models
    |  |- anomaly_scores
    |  |- utils
    |-test

The models themselves are defined in `src/models`. Each file contains a specific
model and should implement the interface below in order to guarantee that each
functions defined in `src/anomaly_scores` or `src/utils` can be called with any
of them. For example `sample(m::AbstractGN)` should be working with every model:

    function sample(model::AbstractGN)
        z = prior_sample(model)
        decoder_sample(model, z)
    end

## Interface 

Every model has to be a subtype of `AbstractGN`. Feel free to add abstract types
like `AbstractVAE <: AbstractGN` if it suits your needs.

    encoder_mean(m::Model, x::AbstractArray)::AbstractArray
    encoder_variance(m::Model, x::AbstractArray)::AbstractArray
    encoder_mean_var(m::Model, x::AbstractArray)::Tuple{AbstractArray, AbstractArray}
    encoder_sample(m::Model, x::AbstractArray)::AbstractArray
    encoder_loglikelihood(m::Model, z::AbstractArray)::Real
    
    prior_mean(m::Model)::AbstractArray
    prior_variance(m::Model)::AbstractArray
    prior_mean_var(m::Model)::Tuple{AbstractArray, AbstractArray}
    prior_sample(m::Model)::AbstractArray
    prior_loglikelihood(m::Model, z::AbstractArray)::Real
    
    decoder_mean(m::Model, z::AbstractArray)::AbstractArray
    decoder_variance(m::Model, z::AbstractArray)::AbstractArray
    decoder_mean_var(m::Model, z::AbstractArray)::Tuple{AbstractArray, AbstractArray}
    decoder_sample(m::Model, z::AbstractArray)::AbstractArray
    decoder_loglikelihood(m::Model, x::AbstractArray, z::AbstractArray)::Real
    
    elbo(m::Model, x::AbstractArray)
