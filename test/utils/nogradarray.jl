@testset "utils/nogradarray.jl" begin

    @info "Testing NoGradArray"

    x = NoGradArray(ones(3))
    y = ones(3)

    p = Gaussian(x, y)
    @test length(params(p)) == 1

    g = gpu(p)
    @test length(params(g)) == 1
    @test rand(g) isa CuArray

    c = cpu(g)
    @test length(params(c)) == 1
    @test rand(c) isa Array

end
