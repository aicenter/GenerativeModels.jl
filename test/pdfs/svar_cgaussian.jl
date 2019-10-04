@testset "pdfs/svar_gaussian.jl" begin

    @info "Testing SharedVarCGaussian"

    xlen  = 3
    zlen  = 2
    batch = 10
    T     = Float32

    mapping = Dense(zlen, xlen)
    var = param(ones(T, xlen))
    p  = SharedVarCGaussian(xlen, zlen, mapping, var) |> gpu
    z  = randn(T, zlen, batch) |> gpu
    μx = mean(p, z)
    σ2 = variance(p)
    x  = rand(p, z)

    @test size(μx) == (xlen, batch)
    @test size(σ2) == size(var)
    @test size(x) == (xlen, batch)
    @test size(loglikelihood(p, x, z)) == (1, batch)

    q  = Gaussian(zeros(T, xlen), ones(T, xlen)) |> gpu
    @test size(kld(p, q, z)) == (1, batch)


    # Test simple function mapping constructor
    p = SharedVarCGaussian(xlen, xlen, x->tanh.(x), var)
    @test isa(p, SharedVarCGaussian{T})

    # Test show function
    msg = @capture_out show(p)
    @test occursin("SharedVarCGaussian", msg)

end
