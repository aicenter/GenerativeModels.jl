@testset "src/models/vamp.jl" begin
	# test different constructors
	xdim, zdim, n, K = 4, 2, 6, 2
	v = VAMP(K, zeros(Float32, xdim, K))
	@test length(params(v)) == 1

	# test gpu compatibility
	v = VAMP(K, zeros(Float32, xdim, K)) |> gpu
	@test typeof(v.pseudoinputs) == typeof(gpu(zeros(Float32,2,2))) # works even when gpu not available

    # also test it as a part of a VAE model, test trainability
	x = f32(hcat(ones(xdim, n), -ones(xdim, n)) + randn(xdim, n*2)/15) |> gpu
	hdim = 50
	enc = f32(Chain(Dense(xdim, hdim, relu), Dense(hdim, zdim*2)))
	dec = f32(Chain(Dense(zdim, hdim, relu), Dense(hdim, xdim+1)))
	q = CMeanVarGaussian{DiagVar}(enc)
	p = CMeanVarGaussian{ScalarVar}(dec)
    v = VAMP(K, zeros(Float32, xdim, K))
	m = VAE(v, q, p) |> gpu # gpu will make a copy of v
	@test typeof(m.prior.pseudoinputs) == typeof(gpu(zeros(Float32,2,2))) # works even when gpu not available
	
	# extract original param vals
    vamp_params_init = get_params(m.prior)
    model_params_init = get_params(m)
    @test length(model_params_init) == 9

    # test trainability
	k = IMQKernel{Float32}(0.1)    
    λ = .1f0
    lossf(x) = λ*mmd_mean_vamp(m, x, k) .- mean(logpdf(m.decoder, x, rand(m.encoder, x)))
    opt = ADAM(0.0001)
	data = [(x,) for _ in 1:2];
    Flux.train!(lossf, params(m), data, opt)
    @test all(param_change(vamp_params_init, m.prior)) # did the vamp params change?
    @test all(param_change(model_params_init, m)) # did the model params change?
end