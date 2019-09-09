@testset "models/vae.jl" begin

    @info "Testing VAE"

    ω0 = 0.5f0
    dt = 0.3f0
    xsize = 30
    zsize = 8
    batch = 20
    noise = 0.01f0

    test_data = randn(xsize, batch)
    
    μe  = Dense(xsize, zsize)
    σ2e = param(ones(zsize))
    encoder = CGaussian(μe, σ2e)

    μd, _ = make_ode_decoder(xsize, (0f0,xsize*dt), 2)
    σ2d   = param(ones(1))
    decoder = CGaussian(μd, σ2d)
    model = VAE(xsize, zsize, encoder, decoder)

    loss = elbo(model, test_data)
    ps = params(model)
    @test length(ps) > 0
    @test isa(loss, Tracker.TrackedReal)
    
end
