@testset "utils/nogradarray.jl" begin

    @info "Testing NoGradArray"

    x = NoGradArray(ones(3))
    y = ones(3)

    p = Gaussian(x, y)
    g = gpu(p)

    @test length(params(p)) == 1
    @test length(params(g)) == 1
end
