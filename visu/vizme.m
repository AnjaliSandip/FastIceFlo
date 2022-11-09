
load JKS2e4
md.miscellaneous.name = 'output';
md=loadresultsfromdisk(md, 'output.outbin')
plotmodel(md,'data',sqrt(md.results.PTsolution.Vx.^2 + md.results.PTsolution.Vy.^2));
