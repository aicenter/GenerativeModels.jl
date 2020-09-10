@testset "VAMP" begin
    # test different constructors
    xdim, zdim, n, K = 4, 2, 6, 2
    v = VAMP(zeros(Float32, xdim, K))
    @test length(Flux.params(v)) == 1
    v = VAMP(K, xdim)
    @test length(Flux.params(v)) == 1

    # test gpu compatibility
    cutype = typeof(gpu(zeros(Float32,2,2)))
    v = VAMP(zeros(Float32, xdim, K)) |> gpu
    @test typeof(v.pseudoinputs) == cutype # works even when gpu not available

    # test constructors
    x = f32(hcat(ones(xdim, n), -ones(xdim, n)) + randn(xdim, n*2)/15) |> gpu

    # mean constructor - K slightly perturbed means of X
    v = init_vamp_mean(3, cpu(x), 0.1f0)
    @test size(v.pseudoinputs) == (xdim,3)
    @test typeof(v.pseudoinputs) == Array{Float32,2}
    v = init_vamp_mean(3, cpu(x)) |> gpu
    @test typeof(v.pseudoinputs) == cutype

    # rand constructor - randomly selects K samples from X
    v = init_vamp_sample(3, cpu(x))
    @test size(v.pseudoinputs) == (xdim,3)
    @test typeof(v.pseudoinputs) == Array{Float32,2}
    v = init_vamp_sample(3, cpu(x)) |> gpu
    @test typeof(v.pseudoinputs) == cutype

    # also test it as a part of a VAE model, test trainability
    hdim = 50
    enc = f32(Chain(Dense(xdim, hdim, relu), SplitLayer(hdim, [zdim,zdim])))
    dec = f32(Chain(Dense(zdim, hdim, relu), SplitLayer(hdim, [xdim,1])))
    q = ConditionalMvNormal(enc)
    p = ConditionalMvNormal(dec)
    v = init_vamp_mean(K, cpu(x), 0.1f0)
    m = VAE(v, q, p) |> gpu # gpu will make a copy of v
    @test typeof(m.prior.pseudoinputs) == typeof(gpu(zeros(Float32,2,2))) # works even when gpu not available

    # extract original param vals
    vamp_params_init = get_params(m.prior)
    model_params_init = get_params(m)
    @test length(model_params_init) == 13

    # test trainability
    k = IMQKernel(0.1f0)
    λ = .1f0
    lossf(x) = λ*mmd_mean(m, x, k) .- mean(logpdf(m.decoder, x, rand(m.encoder, x)))
    opt = ADAM(0.0001)
    data = [(x,) for _ in 1:2];
    Flux.train!(lossf, Flux.params(m), data, opt)
    @test all(param_change(vamp_params_init, m.prior)) # did the vamp params change?
    @test all(param_change(model_params_init, m)) # did the model params change?
end
