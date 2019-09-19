@testset "src/gaussian.jl" begin

    @info "Testing Gaussian"

    p  = Gaussian(zeros(2), ones(2))
    μ  = mean(p)
    σ2 = variance(p)
    @test mean_var(p) == (μ, σ2)
    @test size(rand(p, 10)) == (2, 10)
    @test size(loglikelihood(p, randn(2, 10))) == (1, 10)
    @test size(loglikelihood(p, randn(2))) == (1,)

    q = Gaussian(zeros(2), ones(2))
    @test kld(p,q)[1] == 0.0

    msg = @capture_out show(p)
    @test occursin("Gaussian", msg)

end
