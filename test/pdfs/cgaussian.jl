@testset "src/cgaussian.jl" begin

    @info "Testing CGaussian"

    xlen = 3
    zlen = 2
    batch = 10
    T     = Float64
    V     = ScalarVar

    p  = CGaussian{T,V}(xlen, zlen, Dense(zlen, xlen+1))
    z  = randn(T, zlen, batch)
    μx = mean(p, z)
    σ2 = variance(p, z)

    @test isa(μx, TrackedArray{T})
    @test isa(σ2, TrackedArray{T})
    @test mean_var(p, z) == (μx, σ2)

    @test size(μx) == (xlen, batch)
    @test size(σ2) == (1, batch)
    @test mean_var(p, z) == (μx, σ2)
    @test size(rand(p, z)) == (xlen, batch)

    x = randn(xlen, batch)
    @test size(loglikelihood(p, x, z)) == (1, batch)

    x = randn(xlen)
    z = randn(zlen)
    @test size(loglikelihood(p, x, z)) == (1, 1)

    q = Gaussian(zeros(T, xlen), ones(T, xlen))
    @test size(kld(p,q,z)) == (1, 1)


    V  = DiagVar
    p  = CGaussian{T,V}(xlen, zlen, Dense(zlen, xlen*2))
    z  = randn(T, zlen, batch)
    x  = randn(T, xlen, batch)
    μx = mean(p, z)
    σ2 = variance(p, z)
    q  = Gaussian(zeros(T, xlen), ones(T, xlen))

    @test size(μx) == (xlen, batch)
    @test size(σ2) == (xlen, batch)
    @test size(rand(p, z)) == (xlen, batch)
    @test size(loglikelihood(p, x, z)) == (1, batch)
    @test size(kld(p, q, z)) == (1, batch)


    V  = UnitVar
    p  = CGaussian{T,V}(xlen, zlen, Dense(zlen, xlen))
    z  = randn(T, zlen, batch)
    x  = randn(T, xlen, batch)
    μx = mean(p, z)
    σ2 = variance(p, z)
    q  = Gaussian(zeros(T, xlen), ones(T, xlen))

    @test size(μx) == (xlen, batch)
    @test σ2 == ones(T, xlen)
    @test size(rand(p, z)) == (xlen, batch)
    @test size(loglikelihood(p, x, z)) == (1, batch)
    @test size(kld(p, q, z)) == (1, batch)

    msg = @capture_out show(p)
    @test occursin("CGaussian", msg)

end
