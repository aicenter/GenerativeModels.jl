@testset "src/cgaussian.jl" begin

    @info "Testing CMeanVarGaussian"

    xlen = 3
    zlen = 2
    batch = 10
    T     = Float32

    # Test ScalarVar
    @testset "ScalarVar" begin
        p = CMeanVarGaussian{T,ScalarVar}(f32(Dense(zlen, xlen+1))) |> gpu

        z  = randn(T, zlen, batch) |> gpu
        μx = mean(p, z)
        σ2 = variance(p, z)
        @test mean_var(p, z) == (μx, σ2)
        @test size(μx) == (xlen, batch)
        @test size(σ2) == (1, batch)
        @test size(rand(p, z)) == (xlen, batch)

        x = randn(xlen, batch) |> gpu
        @test size(loglikelihood(p, x, z)) == (1, batch)

        x = randn(xlen) |> gpu
        z = randn(zlen) |> gpu
        @test size(loglikelihood(p, x, z)) == (1, 1)

        q = Gaussian(zeros(T, xlen), ones(T, xlen)) |> gpu
        @test size(kld(p,q,z)) == (1, 1)
    end


    @testset "DiagVar" begin
        p = CMeanVarGaussian{T,DiagVar}(f32(Dense(zlen, xlen*2))) |> gpu

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
    end

    # Test simple function mapping constructor
    p = CMeanVarGaussian{Float32,ScalarVar}(x->tanh.(x))

    # Test show function
    msg = @capture_out show(p)
    @test occursin("CMeanVarGaussian", msg)

end
