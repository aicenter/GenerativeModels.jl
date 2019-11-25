@testset "pdfs/abstract_cvmf.jl" begin
    struct CVMF{T<:Real} <: GenerativeModels.AbstractCVMF{T} end

    cvmf = CVMF{Float32}()
    x = ones(1)

    @test_throws ErrorException mean_var(cvmf, x)
    @test_throws ErrorException concentration(cvmf, x)
    @test_throws ErrorException mean(cvmf, x)
end