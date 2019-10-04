@testset "models/vae.jl" begin

    Random.seed!(1)

    @info "Testing VAE"

    ω0 = 0.5
    dt = 0.3f0
    xlen = 30
    zlen = 8
    batch = 20
    noise = 0.01
    T = Float32

    # TODO: ODE decoder currently not working on GPU
    test_data = randn(T, xlen, batch)
    μe  = Dense(xlen, zlen)
    enc = CGaussian{T,UnitVar}(zlen, xlen, μe)

    μd, _ = make_ode_decoder(xlen, (0f0, xlen*dt), 2)
    dec   = CGaussian{T,UnitVar}(xlen, zlen, μd)
    model = VAE(enc, dec)

    loss = elbo(model, test_data)
    ps = params(model)
    @test Base.length(ps) > 0
    @test isa(loss, Tracker.TrackedReal)

    prior = Gaussian(param(zeros(T, zlen)), param(ones(T, zlen)))
    model = VAE(prior, enc, dec)
    loss = elbo(model, test_data)
    ps = params(model)
    @test Base.length(ps) > 0
    @test isa(loss, Tracker.TrackedReal)

    # simple test for vanilla vae
    xlen = 4
    zlen = 2
    test_data = hcat(ones(T,xlen,Int(batch/2)), -ones(T,xlen,Int(batch/2))) |> gpu

    enc = GenerativeModels.ae_layer_builder([xlen, 10, 10, zlen], relu, Dense)
    enc_dist = CGaussian(zlen, xlen, enc)

    dec = GenerativeModels.ae_layer_builder([zlen, 10, 10, xlen], relu, Dense)
    dec_dist = CGaussian(xlen, zlen, dec)

    model = VAE(enc_dist, dec_dist) |> gpu

    loss = elbo(model, test_data)
    ps = params(model)
    @test Base.length(ps) > 0
    @test isa(loss, Tracker.TrackedReal)

    zs = rand(model.encoder, test_data)
    @test size(zs) == (zlen, batch)
    xs = rand(model.decoder, zs)
    @test size(xs) == (xlen, batch)     

    # test training
    params_init = get_params(model)
    opt = ADAM()
    cb(model, data, loss, opt) = nothing
    data = [(test_data,) for i in 1:10000]
    lossf(x) = elbo(model, x, β=1e-3)
    Flux.train!(lossf, params(model), data, opt)

    @test all(param_change(params_init, model)) # did the params change?
    zs = rand(model.encoder, test_data)
    xs = mean(model.decoder, zs)
    @test all(abs.(test_data - xs) .< 0.2) # is the reconstruction ok?

    # wasserstein vae
    enc = GenerativeModels.ae_layer_builder([xlen, 10, 10, zlen], relu, Dense)
    enc_dist = CGaussian{T,UnitVar}(zlen, xlen, enc)

    dec = GenerativeModels.ae_layer_builder([zlen, 10, 10, xlen+1], relu, Dense)
    dec_dist = CGaussian{T,ScalarVar}(xlen, zlen, dec)

    model = VAE(enc_dist, dec_dist) |> gpu

    # test training
    params_init = get_params(model)
    opt = ADAM()
    cb(model, data, loss, opt) = nothing
    data = [test_data for i in 1:10000];
    k(x,y) = GenerativeModels.imq(x,y,1.0);
    function mse(ŷ, y)
        d = ŷ .- y
        sum(d .* d) * 1 // length(y)
    end
    lossf(x) = mse(x, mean(model.decoder, mean(model.encoder,x))) + mmd(model, x, k)
    train!(model, data, lossf, opt, cb)

    # this works well but has quite a large variance
    @test all(param_change(params_init, model)) 
    zs = mean(model.encoder, test_data)
    xs = mean(model.decoder, zs)
    @test all(abs.(test_data - xs) .< 0.2) 

    msg = @capture_out show(model)
    @test occursin("VAE", msg)
    Random.seed!()  # reset the seed
end
