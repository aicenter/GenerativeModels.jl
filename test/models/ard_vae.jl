@testset "models/ard_vae.jl" begin

    @info "Testing ARDVAE"

    ω0 = 0.5f0
    dt = 0.3f0
    xsize = 30
    zsize = 8
    batch = 20
    noise = 0.01f0
    
    function generate(ω1)
        U, T, Omega = GenerativeModels.generate_sine_data(
            batch, steps=xsize, dt=dt, freq_range=[ω0, ω1])
        U .+= randn(Float32, size(U)) * noise
        return U, T, Omega
    end
    
    encoder = Chain(
         Dense(xsize, 50, σ),
         Dense(50, 50, σ),
         Dense(50, zsize))
    decoder, _ = make_ode_decoder(xsize, (0f0,xsize*dt), 2)
    
    test_data = generate(2.0)[1]
    
    model = ARDVAE{Float32}(xsize, zsize, encoder, decoder)
    loss = elbo(model, test_data)
    ps = params(model)
    @test length(ps) > 0
    @test isa(loss, Tracker.TrackedReal)
    
    (μ0, λz) = prior_mean_var(model)
    @test μ0 == UniformScaling(0)
    @test size(λz) == (zsize,)

    (μz, σz) = encoder_mean_var(model, test_data)
    @test size(μz) == (zsize, batch)
    @test size(σz) == (zsize,)

    (μx, σe) = decoder_mean_var(model, μz)
    @test size(μx) == (xsize, batch)
    @test size(σe) == (1,)

end
