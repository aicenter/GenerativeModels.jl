import Distances: KLDivergence, kl_divergence
export KLDivergence, kl_divergence

const TuMvNormal = Union{DistributionsAD.TuringDenseMvNormal,
                         DistributionsAD.TuringDiagMvNormal,
                         DistributionsAD.TuringScalMvNormal}


function _kld_gaussian(μ1::AbstractArray, σ1::AbstractArray, μ2::AbstractArray, σ2::AbstractArray)
    k  = size(μ1, 1)
    m1 = sum(σ1 ./ σ2, dims=1)
    m2 = sum((μ2 .- μ1).^2 ./ σ2, dims=1)
    m3 = sum(log.(σ2 ./ σ1), dims=1)
    kld = (m1 .+ m2 .+ m3 .- k) ./ 2
end

function _kld_gaussian(p, q)
    (μ1, σ1) = mean(p), var(p)
    (μ2, σ2) = mean(q), var(q)
    _kld_gaussian(μ1, σ1, μ2, σ2)
end

(m::KLDivergence)(p::MvNormal, q::MvNormal) = _kld_gaussian(p,q)
(m::KLDivergence)(p::TuMvNormal, q::TuMvNormal) = _kld_gaussian(p,q)
(m::KLDivergence)(p::ConditionalDists.BMN, q::MvNormal) = _kld_gaussian(p,q)
(m::KLDivergence)(p::ConditionalDists.BMN, q::TuMvNormal) = _kld_gaussian(p,q)

kl_divergence(p,q) = KLDivergence()(p,q)
# kl_divergence(p::MvNormal, q::MvNormal) = KLDivergence()(p,q)
# kl_divergence(p::TuMvNormal, q::TuMvNormal) = KLDivergence()(p,q)
# kl_divergence(p::ConditionalDists.BMN, q::MvNormal) = KLDivergence()(p,q)


function (m::KLDivergence)(p::ConditionalMvNormal, q::MvNormal, z::AbstractArray)
    _kld_gaussian(condition(p,z), q)
end
function (m::KLDivergence)(p::ConditionalMvNormal, q::TuMvNormal, z::AbstractArray)
    _kld_gaussian(condition(p,z), q)
end

kl_divergence(p::ConditionalMvNormal, q::MvNormal, z::AbstractArray) = KLDivergence()(p,q,z)
kl_divergence(p::ConditionalMvNormal, q::TuMvNormal, z::AbstractArray) = KLDivergence()(p,q,z)
