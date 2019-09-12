@testset "pdfs/svar_gaussian.jl" begin

    @info "Testing SharedVarCGaussian"

    xlen  = 3
    zlen  = 2
    batch = 10
    T     = Float32

    mapping = Dense(zlen, xlen)
    var = param(ones(T, xlen))
    p  = SharedVarCGaussian{T}(xlen, zlen, mapping, var)
    z  = randn(T, zlen, batch)
    μx = mean(p, z)
    σ2 = variance(p)
    x  = rand(p, z)

    @test size(μx) == (xlen, batch)
    @test size(σ2) == size(var)
    @test size(x) == (xlen, batch)
    @test size(loglikelihood(p, x, z)) == (1, batch)

    q  = Gaussian(zeros(T, xlen), ones(T, xlen))
    @test size(kld(p, q, z)) == (1, batch)

end
