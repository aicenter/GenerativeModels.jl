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
of them.

    """`ARDAutoEncoder(encoder, decoder)`
    
    AutoEncoder that enforces sparsity on the latent layer.
    """
    function ARDAutoEncoder(xsize::Int, zsize::Int, encoder, decoder)
        ...
    end
    
    
    
    """`encoder_params(m::ARDAutoEncoder, x::AbstractArray)`
    
    Return a tuple of the parameters of the latent distribution.
    """
    function encoder_params(m::ARDAutoEncoder, x::AbstractArray)
        ...
    end
    
    
    """`encoder_sample(m::ARDAutoEncoder, x::AbstractArray)`
    
    Sample from the latent layer. Returns a vector
    """
    function encoder_sample(m::ARDAutoEncoder{T}, x::AbstractArray) where T
        ...
    end
    
    
    """`decode(m::ARDAutoEncoder, z::AbstractArray)`
    
    Reconstruct from a (sampled) latent vector. Returns a matrix
    """
    function decode(m::ARDAutoEncoder, z::AbstractArray)
        ...
    end
    
    
    """`elbo(m::ARDAutoEncoder, x::AbstractArray)`
    
    Computes variational lower bound. Returns a scalar
    """
    function elbo(m::ARDAutoEncoder, x::AbstractArray)
        ...
    end
    
    
    """`loglikelihood(m::ARDAutoEncoder, x::AbstractArray, z::AbstractArray)'
    
    Computes the log-likelihood of the data.
    """
    function loglikelihood(m::ARDAutoEncoder, x::AbstractArray, z::AbstractArray)
        ...
    end
    
    
    """`loglatent(m::ARDAutoEncoder, z::AbstractArray)`
    
    Compute the probability of a latent vector.
    """
    function loglatent(m::ARDAutoEncoder, z::AbstractArray)
        ...
    end
