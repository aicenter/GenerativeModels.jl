@testset "models/ard_vae.jl" begin

    @info "Testing ARDVAE"

    ω0 = 0.5f0
    dt = 0.3f0
    xsize = 30
    zsize = 8
    batch = 20
    noise = 0.01f0
    
    test_data = randn(Float32, xsize, batch)
    
    encoder = Dense(xsize, zsize)
    decoder, _ = make_ode_decoder(xsize, (0f0,xsize*dt), 2)
    model = ARDVAE{Float32}(xsize, zsize, encoder, decoder)

    loss = elbo(model, test_data)
    ps = params(model)
    @test length(ps) > 0
    @test isa(loss, Tracker.TrackedReal)
    
    (μ0, λz) = prior_mean_var(model)
    @test μ0 == zeros(Float32, zsize)
    @test size(λz) == (zsize,)

    (μz, σz) = encoder_mean_var(model, test_data)
    z = encoder_sample(model, test_data)
    @test size(μz) == (zsize, batch)
    @test size(σz) == (zsize,)
    @test size(z) == (zsize, batch)

    (μx, σe) = decoder_mean_var(model, μz)
    xrec = decoder_sample(model, μz)
    @test size(μx) == (xsize, batch)
    @test size(σe) == (1,)
    @test size(xrec) == (xsize, batch)

    llh = decoder_loglikelihood(model, test_data, z)
    @test size(llh) == (batch,)

    llh = decoder_loglikelihood(model, test_data[:,1], z[:,1])
    @test size(llh) == ()
end
