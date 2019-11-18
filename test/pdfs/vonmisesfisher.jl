@testset "src/vonmisesfisher.jl" begin

    T = Float32

    p  = VonMisesFisher(T.([1, 0, 0]), T(1))
    μ  = mean(p)
    κ = concentration(p)
    @test mean_conc(p) == (μ, κ)
    @test size(rand(p, 10)) == (3, 10)
    @test size(loglikelihood(p, randn(T, 3, 10))) == (1, 10)
    @test size(loglikelihood(p, randn(T, 3))) == (1, 1)
    @test length(Flux.trainable(p)) == 1

    @test eltype(loglikelihood(p, randn(T, 3, 10))) == T
    @test eltype(rand(p, 10)) == T

    q = VonMisesFisher(zeros(2), ones(1))
    @test length(Flux.trainable(q)) == 2
    @test size(kld(q, HypersphericalUniform{Float64}(2))) == ()

    μ = NoGradArray(zeros(2))
    p = VonMisesFisher(μ, ones(1))
    @test length(Flux.trainable(p)) == 1

    p  = VonMisesFisher(NoGradArray([1, 0, 0.]), NoGradArray([1.0]))
    @test length(Flux.trainable(p)) == 0

    # msg = @capture_out show(p)
    # @test occursin("VonMisesFisher", msg)
end