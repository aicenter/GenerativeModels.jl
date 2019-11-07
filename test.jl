using Flux
using Revise
using GenerativeModels

slen = 2
tlen = 10
tspan = (0,10)

m = Chain(NALU(slen,slen), NALU(slen,slen))
d = FluxODEDecoder(slen,tlen,tspan,m)

z = GenerativeModels.destructure(m)
z = randn(length(z) + slen)
d(z)
