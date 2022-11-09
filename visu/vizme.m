
load example
md.miscellaneous.name = 'example';
md=loadresultsfromdisk(md, 'example.outbin')
plotmodel(md,'data',sqrt(md.results.PTsolution.Vx.^2 + md.results.PTsolution.Vy.^2));
