using Test
using Logging
using Parameters
using Random
using GenerativeModels


@testset "models/gan.jl" begin

    Random.seed!(1)

    @info "Testing GAN"
    xlen = 4
    zlen = 2
    batch = 20
    T = Float64

    test_data = hcat(ones(T,xlen,Int(batch/2)), -ones(T,xlen,Int(batch/2)))

    gen = GenerativeModels.ae_layer_builder([zlen, 10, 10, xlen], relu, Dense)
    gen_dist = CGaussian{T,UnitVar}(xlen, zlen, gen)

    disc = GenerativeModels.ae_layer_builder([xlen, 10, 10, 1], relu, Dense)
    disc_dist = CGaussian{T,UnitVar}(1, xlen, disc)

    model = GAN(gen_dist, disc_dist)

    zs = rand(model.prior, batch)
    @test size(zs) == (zlen, batch)
    xs = mean(model.generator, zs)
    @test size(xs) == (xlen, batch)
    dgs = mean(model.discriminator, xs)
    @test size(dgs) == (1, batch)
    dds = mean(model.discriminator, test_data)
    @test size(dds) == (1, batch)

    Random.seed!()
end
