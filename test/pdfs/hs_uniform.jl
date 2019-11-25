@testset "src/hs_uniform.jl" begin
    
    hu = HypersphericalUniform(3)
    @test length(hu) == 3
    @test size(rand(hu, 5), 2) == 5
    
    r = rand(hu, 5)
    for i in 1:size(r, 2)
        @test norm(r[:, i]) ≈ 1
    end
end