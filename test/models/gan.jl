@testset "models/gan.jl" begin

    Random.seed!(1)

    @info "Testing GAN"
    xlen = 4
    zlen = 2
    batchsize = 20
    T = Float32

    test_data = hcat(ones(T,xlen,Int(batchsize/2)), -ones(T,xlen,Int(batchsize/2))) |> gpu

    gen = GenerativeModels.ae_layer_builder([zlen, 10, 10, xlen], relu, Dense)
    gen_dist = CGaussian{T,UnitVar}(xlen, zlen, gen)

    disc = GenerativeModels.ae_layer_builder([xlen, 10, 10, 1], relu, Dense; last = Flux.σ)
    disc_dist = CGaussian{T,UnitVar}(1, xlen, disc)

    model = GAN(gen_dist, disc_dist) |> gpu

    zs = rand(model.prior, batchsize)
    @test size(zs) == (zlen, batchsize)
    display(zs)
    xs = mean(model.generator, zs)
    @test size(xs) == (xlen, batchsize)
    dgs = mean(model.discriminator, xs)
    @test size(dgs) == (1, batchsize)
    dds = mean(model.discriminator, test_data)
    @test size(dds) == (1, batchsize)

    # test parameters
    ps = params(model)
    @test Base.length(ps) == 12

    # # test losses
    # lg = generator_loss(model, zs)
    # @test isa(lg, Real)
    # lg = generator_loss(model, batchsize)
    # @test isa(lg, Real)
    # 
    # ld = discriminator_loss(model, test_data, zs)
    # @test isa(ld, Real)
    # ld = discriminator_loss(model, test_data)
    # @test isa(ld, Real)

    # #  are discriminator and generator are only trained with their losses?
    # # test generator loss
    # params_gen = get_params(model.generator)
    # params_disc = get_params(model.discriminator)
    # gloss(x) = generator_loss(model, batchsize) # gen_loss does not need input data
    # opt = ADAM()
    # GenerativeModels.loss_back_update!(model, test_data, gloss, opt)
    # @test all(param_change(params_gen, model.generator))
    # @test !any(param_change(params_disc, model.discriminator))

    # # test discriminator loss
    # params_gen = get_params(model.generator)
    # params_disc = get_params(model.discriminator)
    # dloss(x) = discriminator_loss(model, x)
    # opt = ADAM()
    # GenerativeModels.loss_back_update!(model, test_data, dloss, opt)
    # @test all(param_change(params_disc, model.discriminator))
    # @test !any(param_change(params_gen, model.generator))
    # 
    # Random.seed!()
end
