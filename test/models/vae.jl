@testset "models/vae.jl" begin

    @info "Testing VAE"

    ω0 = 0.5
    dt = 0.3
    xsize = 30
    zsize = 8
    batch = 20
    noise = 0.01
    T = Float64

    test_data = randn(xsize, batch)
    
    μe  = Dense(xsize, zsize)
    encoder = CGaussian{T,UnitVar}(zsize, xsize, μe)

    μd, _ = make_ode_decoder(xsize, (0., xsize*dt), 2)
    decoder = CGaussian{T,UnitVar}(xsize, zsize, μd)
    model = VAE(xsize, zsize, encoder, decoder)

    loss = elbo(model, test_data)
    ps = params(model)
    @test Base.length(ps) > 0
    @test isa(loss, Tracker.TrackedReal)
    
end
