using Flux
using GenerativeModels

slen = 2; tlen=5; dt=2π/tlen;
z = Float32.([0, 1, -1, 0, 0, 0, 0, 1]);

rodent = Rodent(slen, tlen, dt, enc)
display(rodent)

display(reshape(mean(rodent.decoder, z), slen, tlen))
display(reshape(rodent.decoder.mapping(z), slen,tlen))
