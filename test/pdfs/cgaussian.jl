@testset "src/cgaussian.jl" begin

    @info "Testing CGaussian"

    xlength = 3
    zlength = 2
    batch   = 10
    T       = Float64
    V       = ScalarVar

    p  = CGaussian{T,V}(xlength, zlength, Dense(zlength, xlength+1))
    z  = randn(T, zlength, batch)
    μx = mean(p, z)
    σ2 = variance(p, z)

    @test isa(μx, TrackedArray{T})
    @test isa(σ2, TrackedArray{T})
    @test mean_var(p, z) == (μx, σ2)

    @test size(μx) == (xlength, batch)
    @test size(σ2) == (1, batch)
    @test mean_var(p, z) == (μx, σ2)
    @test size(sample(p, z, batch=10)) == (xlength, batch)

    x = randn(xlength, batch)
    @test size(loglikelihood(p, x, z)) == (1, batch)

    x = randn(xlength)
    z = randn(zlength)
    @test size(loglikelihood(p, x, z)) == (1, 1)

    q = Gaussian(zeros(T, xlength), ones(T, xlength))
    @test size(kld(p,q,z)) == (1, 1)


    V  = DiagVar
    p  = CGaussian{T,V}(xlength, zlength, Dense(zlength, xlength*2))
    z  = randn(T, zlength, batch)
    x  = randn(T, xlength, batch)
    μx = mean(p, z)
    σ2 = variance(p, z)
    q  = Gaussian(zeros(T, xlength), ones(T, xlength))

    @test size(μx) == (xlength, batch)
    @test size(σ2) == (xlength, batch)
    @test size(sample(p, z, batch=10)) == (xlength, batch)
    @test size(loglikelihood(p, x, z)) == (1, batch)
    @test size(kld(p, q, z)) == (1, batch)


    V  = UnitVar
    p  = CGaussian{T,V}(xlength, zlength, Dense(zlength, xlength))
    z  = randn(T, zlength, batch)
    x  = randn(T, xlength, batch)
    μx = mean(p, z)
    σ2 = variance(p, z)
    q  = Gaussian(zeros(T, xlength), ones(T, xlength))

    @test size(μx) == (xlength, batch)
    @test σ2 == I
    @test size(sample(p, z, batch=10)) == (xlength, batch)
    @test size(loglikelihood(p, x, z)) == (1, batch)
    @test size(kld(p, q, z)) == (1, batch)

end
