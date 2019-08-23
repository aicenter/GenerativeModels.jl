@testset "models/ard_autoencoder.jl" begin

    @info "Testing ARD AutoEncoder"

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
    
    init(s...) = randn(Float32, s...) ./ 1000f0
     encoder = Chain(
         Dense(xsize, 50, σ, initW=init),
         Dense(50, 50, σ, initW=init),
         Dense(50, zsize, initW=init))
    decoder, _ = make_order2ode_decoder(xsize, (0f0,xsize*dt))
    
    test_data = generate(2.0)[1]
    
    model = ARDAutoEncoder{Float32}(xsize, zsize, encoder, decoder)
    loss = elbo(model, test_data)
    ps = params(model)
    
    @test isa(loss, Tracker.TrackedReal)
    @test length(ps) > 0
    
    (μz, σz, γ) = GenerativeModels.encoder_params(model, test_data)
    @test size(μz) == (zsize, batch)
    @test size(σz) == (zsize,)
    @test size(γ)  == (zsize,)
    
    xrec = GenerativeModels.decode(model, μz)
    @test size(xrec) == (xsize, batch)

end
