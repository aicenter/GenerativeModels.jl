@testset "src/cgaussian.jl" begin

    @info "Testing CGaussian"

    xlen = 3
    zlen = 2
    batch = 10
    T     = Float32

    # Test ScalarVar
    p = CGaussian(xlen, zlen, f32(Dense(zlen, xlen+1))) |> gpu
    @test isa(p, CGaussian{T,ScalarVar})

    z  = randn(T, zlen, batch) |> gpu
    μx = mean(p, z)
    σ2 = variance(p, z)
    @test isa(μx, TrackedArray{T})
    @test isa(σ2, TrackedArray{T})
    @test mean_var(p, z) == (μx, σ2)
    @test size(μx) == (xlen, batch)
    @test size(σ2) == (1, batch)
    @test mean_var(p, z) == (μx, σ2)
    @test size(rand(p, z)) == (xlen, batch)

    x = randn(xlen, batch) |> gpu
    @test size(loglikelihood(p, x, z)) == (1, batch)

    x = randn(xlen) |> gpu
    z = randn(zlen) |> gpu
    @test size(loglikelihood(p, x, z)) == (1, 1)

    q = Gaussian(zeros(T, xlen), ones(T, xlen)) |> gpu
    @test size(kld(p,q,z)) == (1, 1)


    # Test DiagVar
    p = CGaussian(xlen, zlen, f32(Dense(zlen, xlen*2))) |> gpu
    @test isa(p, CGaussian{T,DiagVar})

    z  = randn(T, zlen, batch) |> gpu
    x  = randn(T, xlen, batch) |> gpu
    μx = mean(p, z)
    σ2 = variance(p, z)
    @test size(μx) == (xlen, batch)
    @test size(σ2) == (xlen, batch)
    @test size(rand(p, z)) == (xlen, batch)
    @test size(loglikelihood(p, x, z)) == (1, batch)

    q = Gaussian(zeros(T, xlen), ones(T, xlen)) |> gpu
    @test size(kld(p, q, z)) == (1, batch)


    # Test UnitVar
    p = CGaussian(xlen, zlen, f32(Dense(zlen, xlen))) |> gpu
    @test isa(p, CGaussian{T,UnitVar})

    z  = randn(T, zlen, batch) |> gpu
    x  = randn(T, xlen, batch) |> gpu
    μx = mean(p, z)
    σ2 = variance(p, z)
    @test size(μx) == (xlen, batch)
    if has_cuda() CuArrays.allowscalar(true) end
    msg = "Performing scalar operations on GPU arrays: "
    msg *= "This is very slow, consider disallowing these operations with `allowscalar(false)`"
    @test (@test_logs (:warn, msg) σ2 == ones(T, xlen))
    if has_cuda() CuArrays.allowscalar(false) end
    @test size(rand(p, z)) == (xlen, batch)
    @test size(loglikelihood(p, x, z)) == (1, batch)

    q  = Gaussian(zeros(T, xlen), ones(T, xlen)) |> gpu
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
