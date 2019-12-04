@testset "pdfs/constspec_gaussian.jl" begin

    xlen  = 3
    zlen  = 2
    batch = 10
    T     = Float32

    @testset "DiagVar" begin
        pdf = Gaussian(zeros(T,xlen), ones(T,xlen)) |> gpu

        mapping = Dense(zlen, xlen)
        var = NoGradArray(ones(T, xlen))
        cpdf = CMeanGaussian{T,DiagVar}(mapping, var) |> gpu

        p = ConstSpecGaussian(pdf, cpdf)
        z  = randn(T, zlen, batch) |> gpu

        μc = const_mean(p)
        σc = const_variance(p)
        @test size(μc) == (xlen,)
        @test size(μc) == (xlen,)

        μs = spec_mean(p, z)
        σs = spec_variance(p, z)
        @test size(μs) == (xlen,batch)
        @test size(σs) == (xlen,batch)

        _μc, _μs = mean(p,z)
        @test all(μc .== _μc)
        @test all(μs .== _μs)
        _σc, _σs = variance(p,z)
        @test all(σc .== _σc)
        @test all(σs .== _σs)

        (c,s) = rand(p, z)
        x = c .+ s
        @test length(params(p)) == 4
        @test size(loglikelihood(p,x,z)) == (1,batch)

        # Test show function
        msg = @capture_out show(p)
        @test occursin("ConstSpecGaussian", msg)

    end

end
