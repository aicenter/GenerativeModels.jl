@testset "models/vae.jl" begin

    Random.seed!(0)

    @testset "Vanilla VAE" begin
        T = Float32
        xlen = 4
        zlen = 2
        batch = 20
        test_data = randn(T, 4, batch)/100 .+ hcat(ones(T,xlen,Int(batch/2)), -ones(T,xlen,Int(batch/2))) |> gpu
    
        enc = GenerativeModels.stack_layers([xlen, 4, zlen], relu, Dense)
        enc_dist = ConditionalMvNormal(enc)
    
        dec = GenerativeModels.stack_layers([zlen, 4, xlen], relu, Dense)
        dec_dist = ConditionalMvNormal(dec)
    
        model = VAE(zlen, enc_dist, dec_dist) |> gpu
    
        loss = - elbo(model, test_data)
        ps = Flux.params(model)
        @test length(ps) > 0
        @test isa(loss, Real)
    
        zs = rand(model.encoder, test_data)
        @test size(zs) == (zlen, batch)
        xs = mean(model.decoder, zs)
        @test size(xs) == (xlen, batch)     
    
        # test training
        params_init = get_params(model)
        opt = ADAM()
        data = [(test_data,) for i in 1:2000]
        lossf(x) = - elbo(model, x, β=1e0)
        Flux.train!(lossf, Flux.params(model), data, opt)
    
        @test all(param_change(params_init, model)) # did the params change?
        nxs = mean(model.decoder, rand(model.encoder, test_data))
        @debug maximum(test_data - nxs)
        @test Flux.mse(test_data, nxs) < Flux.mse(test_data, xs)
        # this seems to be a stable way to test reconstruction
    end

    @testset "Wasserstein VAE" begin

        T = Float32
        xlen = 4
        zlen = 2
        batch = 20
        test_data = hcat(ones(T,xlen,Int(batch/2)), -ones(T,xlen,Int(batch/2))) |> gpu

        enc = GenerativeModels.stack_layers([xlen, 10, 10, zlen], relu, Dense)
        enc_dist = ConditionalMvNormal(enc)

        dec = GenerativeModels.stack_layers([zlen, 10, 10, xlen], relu, Dense)
        dec_dist = ConditionalMvNormal(dec)

        model = VAE(zlen, enc_dist, dec_dist) |> gpu

        # test training
        params_init = get_params(model)
        opt = ADAM()
        k = GenerativeModels.IMQKernel()
        mmd(x) = GenerativeModels.mmd_mean(model, x, k)
        data = (test_data,)
        lossf(x) = Flux.mse(x, mean(model.decoder, mean(model.encoder,x))) + mmd(x)
        GenerativeModels.update_params!(model, data, lossf, opt)
        ps = Flux.params(model)
        @test all(param_change(params_init, model)) 

        # this works well but has quite a large variance
        data = [(test_data,) for _ in 1:10000]
        Flux.train!(lossf, Flux.params(model), data, opt)
        zs = mean(model.encoder, test_data)
        xs = mean(model.decoder, zs)
        @debug maximum(abs.(test_data - xs))
        @test all(abs.(test_data - xs) .< 0.2) 

        msg = sprint(show, model)
        @test occursin("VAE", msg)
        Random.seed!()  # reset the seed
    end

end
