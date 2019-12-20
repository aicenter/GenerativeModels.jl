@testset "utils/flux_ode_decoder.jl" begin

    slength = 2
    xlength = 10
    dt = 0.1f0

    ode = Dense(slength, slength)
    dec = FluxODEDecoder(slength, xlength, dt, ode)
    ps = GenerativeModels.destructure(ode)
    u0 = rand(slength)
    z  = vcat(ps, u0)

    x = dec(z)
    @test length(z) == 8
    @test length(x) == xlength*slength

    loss(z) = sum(dec(z))
    gs = Flux.gradient(loss, z)

    @test length(gs) == 1
    @test length(gs[1]) == 8

end
