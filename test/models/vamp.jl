using Test
using ConditionalDists
using GenerativeModels
using Flux
using CuArrays

@testset "src/models/vamp.jl" begin
	xdim, zdim, n, K = 4, 2, 6, 2
	v = VAMP(K, zeros(Float32, xdim, K))
	@test length(params(v)) == 1

	x = f32(hcat(ones(xdim, n), -ones(xdim, n)) + randn(xdim, n*2)/15)
	hdim = 50
	enc = f32(Chain(Dense(xdim, hdim, relu), Dense(hdim, zdim*2)))
	dec = f32(Chain(Dense(zdim, hdim, relu), Dense(hdim, xdim+1)))
	q = CMeanVarGaussian{DiagVar}(enc)
	p = CMeanVarGaussian{ScalarVar}(dec)
    v = VAMP(K, zeros(Float32, xdim, K))
	m = VAE(v, q, p)
	
	k = IMQKernel{Float32}(0.1)    
    λ = .1f0
    lossf(x) = λ*mmd_mean_vamp(m, x, k) .- mean(loglikelihood(m.decoder, x, rand(m.encoder, x)))
    opt = ADAM()
	data = [(x,) for _ in 1:1000];
    Flux.train!(lossf, params(m), data, opt)
    rx = mean(m.decoder, mean(m.encoder, x))
    @test all(abs.(rx - x) .< 0.2)
end