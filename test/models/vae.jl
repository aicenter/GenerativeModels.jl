using Test
using Logging

using Revise
using GenerativeModels

@testset "models/vae.jl" begin

    @info "Testing VAE"

    xdim = 4
    zdim = 2
    N = 10
    X = randn(xdim, N)
    T = Float32

    # test the simplest constructor
    encoder = Dense(xdim, zdim*2)
    decoder = Dense(zdim, xdim)
    model_default = VAE(xdim, zdim, encoder, decoder)
    model_unit = VAE{T}(xdim, zdim, encoder, decoder)
    model = VAE{T,UnitVar}(xdim, zdim, encoder, decoder)
    @test typeof(model) == typeof(model_unit)
    @test typeof(model) == typeof(model_default)    
    decoder = Dense(zdim, xdim+1)
    model_scalar = VAE{T,ScalarVar}(xdim, zdim, encoder, decoder)
    @test typeof(model_scalar) != typeof(model_unit)    
    @test typeof(model_scalar) != typeof(model_default)
    decoder = Dense(zdim, xdim*2)
    model = VAE{T,DiagVar}(xdim, zdim, encoder, decoder)
    @test typeof(model) != typeof(model_unit)
    
    # test prior methods
    @test prior_mean(model) == zeros(T,zdim)
    @test eltype(prior_mean(model)) == T
    b = randn(3,3)
    @test b*prior_variance(model) == b*prior_variance(model) == b
    
    # encoder
    @test size(encoder_mean(model, X)) == (zdim,N)
    @test typeof(encoder_mean(model, X)) <: Tracker.TrackedArray
    @test size(encoder_variance(model, X)) == (zdim,N)
    @test typeof(encoder_variance(model, X)) <: Tracker.TrackedArray
    Z = encoder_sample(model, X)
    @test size(Z) == (zdim, N)

    # decoder
    # diag model
    @test size(decoder_mean(model, Z)) == (xdim,N)
    @test typeof(decoder_mean(model, Z)) <: Tracker.TrackedArray
    @test size(decoder_variance(model, Z)) == (xdim,N)
    @test typeof(decoder_variance(model, Z)) <: Tracker.TrackedArray
    # unit model
    @test size(decoder_mean(model_unit, Z)) == (xdim,N)
    @test typeof(decoder_mean(model_unit, Z)) <: Tracker.TrackedArray
    @test decoder_variance(model_unit, Z) == I
    @test !(typeof(decoder_variance(model_unit, Z)) <: Tracker.TrackedArray)
    # scalar model
    @test size(decoder_mean(model_scalar, Z)) == (xdim,N)
    @test typeof(decoder_mean(model_scalar, Z)) <: Tracker.TrackedArray
    @test size(decoder_variance(model_scalar, Z)) == (1,N)
    @test typeof(decoder_variance(model_scalar, Z)) <: Tracker.TrackedArray
end
