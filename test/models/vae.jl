@testset "models/vae.jl" begin

    @info "Testing VAE"

    ω0 = 0.5
    dt = 0.3
    xlen = 30
    zlen = 8
    batch = 20
    noise = 0.01
    T = Float64

    test_data = randn(xlen, batch)
    
    μe  = Dense(xlen, zlen)
    enc = CGaussian{T,UnitVar}(zlen, xlen, μe)

    μd, _ = make_ode_decoder(xlen, (0., xlen*dt), 2)
    dec   = CGaussian{T,UnitVar}(xlen, zlen, μd)
    model = VAE(enc, dec)

    loss = elbo(model, test_data)
    ps = params(model)
    @test Base.length(ps) > 0
    @test isa(loss, Tracker.TrackedReal)

    prior = Gaussian(param(zeros(T, zlen)), param(ones(T, zlen)))
    model = VAE{T}(prior, enc, dec)
    loss = elbo(model, test_data)
    ps = params(model)
    @test Base.length(ps) > 0
    @test isa(loss, Tracker.TrackedReal)

    # simple test for vanilla vae
    enc = GenerativeModels.ae_layer_builder([xlen, 10, 10, zlen*2], relu, Dense)
    enc_dist = CGaussian{T,DiagVar}(zlen, xlen, enc)

    dec = GenerativeModels.ae_layer_builder([zlen, 10, 10, xlen+1], relu, Dense)
    dec_dist = CGaussian{T,ScalarVar}(xlen, zlen, dec)

    model = VAE(enc_dist, dec_dist)

    loss = elbo(model, test_data)
    ps = params(model)
    @test Base.length(ps) > 0
    @test isa(loss, Tracker.TrackedReal)

    zs = sample(model.encoder, test_data)
    @test size(zs) == (zlen, batch)
    xs = sample(model.decoder, zs)
    @test size(xs) == (xlen, batch)     
end
