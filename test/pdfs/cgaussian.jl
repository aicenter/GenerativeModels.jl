@testset "src/cgaussian.jl" begin

    @info "Testing CGaussian"

    xlen = 3
    zlen = 2
    batch = 10
    T     = Float32

    # Test ScalarVar
    p = CGaussian(xlen, zlen, f32(Dense(zlen, xlen+1)))
    @test isa(p, CGaussian{T,ScalarVar})

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


    # Test DiagVar
    p = CGaussian(xlen, zlen, f32(Dense(zlen, xlen*2)))
    @test isa(p, CGaussian{T,DiagVar})

    z  = randn(T, zlen, batch)
    x  = randn(T, xlen, batch)
    μx = mean(p, z)
    σ2 = variance(p, z)
    @test size(μx) == (xlen, batch)
    @test size(σ2) == (xlen, batch)
    @test size(rand(p, z)) == (xlen, batch)
    @test size(loglikelihood(p, x, z)) == (1, batch)

    q = Gaussian(zeros(T, xlen), ones(T, xlen))
    @test size(kld(p, q, z)) == (1, batch)


    # Test UnitVar
    T = Float64
    p = CGaussian(xlen, zlen, f64(Dense(zlen, xlen)))
    @test isa(p, CGaussian{T,UnitVar})

    z  = randn(T, zlen, batch)
    x  = randn(T, xlen, batch)
    μx = mean(p, z)
    σ2 = variance(p, z)
    @test size(μx) == (xlen, batch)
    @test σ2 == ones(T, xlen)
    @test size(rand(p, z)) == (xlen, batch)
    @test size(loglikelihood(p, x, z)) == (1, batch)

    q  = Gaussian(zeros(T, xlen), ones(T, xlen))
    @test size(kld(p, q, z)) == (1, batch)

    msg = @capture_out show(p)
    @test occursin("CGaussian", msg)

    # Test simple function mapping constructor
    p = CGaussian(xlen, xlen, x->tanh.(x))
    @test isa(p, CGaussian{Float32,UnitVar})

    # Test show function
    msg = @capture_out show(p)
    @test occursin("CGaussian", msg)


end
