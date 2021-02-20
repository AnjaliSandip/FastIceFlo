%Choose your mesh
meshid = 2;
switch(meshid)
	case 1 %uniform structured
		md=squaremesh(model(),1000000,1000000,20,20);
	case 2 %uniform unstructured coarse
		md=triangle(model(),'./TestFiles/Square.exp',50000.);
	case 3 %uniform unstructured finer
		md=triangle(model(),'./TestFiles/Square.exp',20000.);
	case 4 %non unfiform unstructured
		hvertices=[100e3;100e3;1e3;100e3];
		md=bamg(md,'domain','TestFiles/Square.exp','hvertices',hvertices);
	otherwise
		error('not supported yet');
end

%Floating ice, Homorgeneous Dirichlet everywhere
md=setmask(md,'all',''); md=parameterize(md,'./TestFiles/SquareShelfConstrained.par');
%Floating ice, Neumann at ice front
md=setmask(md,'all',''); md=parameterize(md,'./TestFiles/SquareShelf.par');
%Grounded ice, Homorgeneous Dirichlet everywhere
md=setmask(md,'','');    md=parameterize(md,'./TestFiles/SquareSheetConstrained.par');
%Grounding line and Neumann at ice front
md=setmask(md,'./TestFiles/SquareShelf.exp',''); md=parameterize(md,'./TestFiles/SquareSheetShelf.par'); pos=find(md.mesh.y<5e5); md.geometry.bed(pos) = md.geometry.base(pos); md=sethydrostaticmask(md);

md.mask.ice_levelset(md.mesh.y>80e4)=+1;

md.stressbalance.restol=1e-10;
md.groundingline.friction_interpolation = 'SubelementFriction1';
md.groundingline.migration = 'SubelementMigration';
md=setflowequation(md,'SSA','all');

%md=solve(md,'sb','batch','yes')



