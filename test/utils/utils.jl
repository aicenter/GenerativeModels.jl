@testset "utils/utils.jl" begin
    @testset "destructure/restructure" begin
        m = Dense(3,3)
        σ = NoGradArray(ones(3))
        g = CMeanGaussian{Float32,DiagVar}(m, σ)

        z = GenerativeModels.destructure(g)
        @test z isa Vector
        @test length(z) == 3*3 + 3 + 3

        _z = rand(3*3 + 3 + 3)
        _g = GenerativeModels.restructure(g, _z)

        @test _g.mapping.W[1] != g.mapping.W[1]
        @test _g.mapping.W[1] == _z[1]
    end
end
