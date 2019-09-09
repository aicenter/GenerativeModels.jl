@testset "src/gaussian.jl" begin

    @info "Testing PDFS"

    p  = Gaussian(zeros(2), ones(2))
    μ  = mean(p)
    σ2 = variance(p)
    @test μ == zeros(2)
    @test σ2 == ones(2)
    @test mean_var(p) == (μ, σ2)
    @test size(sample(p, batch=10)) == (2, 10)
    @test size(loglikelihood(p, randn(2, 10))) == (1, 10)
    @test size(loglikelihood(p, randn(2))) == (1,)


    p  = CGaussian(Dense(2,3), ones(3))
    z  = randn(2, 10)
    μ  = mean(p, z)
    σ2 = variance(p, z)
    @test typeof(μ) <: TrackedArray
    @test typeof(σ2) <: Array
    @test mean_var(p, z) == (μ, σ2)
    @test size(sample(p, z, batch=10)) == (3, 10)

    x = randn(3, 10)
    @test size(loglikelihood(p, x, z)) == (1, 10)

    x = randn(3)
    z = randn(2)
    @test size(loglikelihood(p, x, z)) == (1,)

end
