@testset "pdfs/abstract_pdf.jl" begin
    @info "Testing abstract PDFs"
    struct PDF{T<:Real} <: GenerativeModels.AbstractPDF{T} end
    struct CPDF{T<:Real} <: GenerativeModels.AbstractCPDF{T} end

    pdf = PDF{Float32}()
    cpdf = CPDF{Float32}()
    x = ones(1)

    @test_throws ErrorException mean_var(pdf)
    @test_throws ErrorException rand(pdf)
    @test_throws ErrorException length(pdf)
    @test_throws ErrorException loglikelihood(pdf, x)

    @test_throws ErrorException mean_var(cpdf, x)
    @test_throws ErrorException xlength(cpdf)
    @test_throws ErrorException zlength(cpdf)
    @test_throws ErrorException rand(cpdf, x)
    @test_throws ErrorException loglikelihood(cpdf, x, x)

    @test_throws ErrorException kld(pdf, pdf)
    @test_throws ErrorException kld(cpdf, pdf, x)

    xlen = 2
    @test GenerativeModels.detect_mapping_variant(ones(xlen), xlen) == UnitVar
    @test GenerativeModels.detect_mapping_variant(ones(xlen+1), xlen) == ScalarVar
    @test GenerativeModels.detect_mapping_variant(ones(xlen*2), xlen) == DiagVar
    @test_throws ErrorException GenerativeModels.detect_mapping_variant(ones(xlen+3), xlen)
end
