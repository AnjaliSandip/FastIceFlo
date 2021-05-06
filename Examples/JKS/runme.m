steps=[4];

if any(steps==1)
	disp('	Step 1: Mesh creation'); 
	md=triangle(model,'Domain.exp',2000);
	[lat lon] = xy2ll(md.mesh.x,md.mesh.y,+1,45,70);
	[x2   y2] = ll2xy(lat,lon,+1,39,71);

	%Get observed velocity field on mesh nodes
	ncdata = [issmdir() 'examples/Data/Greenland_5km_dev1.2.nc'];
	if ~exist(ncdata,'file'), 
		error('File not downloaded in Data Directory');
	end
	x1   = ncread(ncdata,'x1');
	y1   = ncread(ncdata,'y1');
	velx = ncread(ncdata,'surfvelx');
	vely = ncread(ncdata,'surfvely');
	vx   = InterpFromGridToMesh(x1,y1,velx',x2,y2,0);
	vy   = InterpFromGridToMesh(x1,y1,vely',x2,y2,0);
	vel  = sqrt(vx.^2+vy.^2);

	%refine mesh using surface velocities as metric
	md=bamg(md,'hmin',1200,'hmax',15000,'field',vel,'err',5);
	[md.mesh.lat,md.mesh.long]  = xy2ll(md.mesh.x,md.mesh.y,+1,45,70);
	
	save JksMesh md
end 
if any(steps==2)
	disp('	Step 2: Parameterization');
	md=loadmodel('JksMesh');
	
	md=setmask(md,'','');
	md=parameterize(md,'Jks.par'); 
	md=setflowequation(md,'SSA','all');

	save JksPar md
end 
if any(steps==3)
	disp('	Step 3: Control method friction');
	md=loadmodel('JksPar');

	% Control general
	md.inversion=m1qn3inversion(md.inversion);
	md.inversion.iscontrol=1;
	md.inversion.maxsteps=40;
	md.inversion.maxiter=40;
	md.inversion.dxmin=0.1;
	md.inversion.gttol=1.0e-6;
	md.verbose=verbose('control',true);

	% Cost functions
	md.inversion.cost_functions=[101 103 501];
	md.inversion.cost_functions_coefficients=ones(md.mesh.numberofvertices,3);
	md.inversion.cost_functions_coefficients(:,1)=40;
	md.inversion.cost_functions_coefficients(:,2)=1;
	md.inversion.cost_functions_coefficients(:,3)=8e-7;

	% Controls
	md.inversion.control_parameters={'FrictionCoefficient'};
	md.inversion.min_parameters=1*ones(md.mesh.numberofvertices,1);
	md.inversion.max_parameters=200*ones(md.mesh.numberofvertices,1);

	% Solve
	md=solve(md,'Stressbalance');

	% Update model friction fields accordingly
	md.friction.coefficient=md.results.StressbalanceSolution.FrictionCoefficient;
	
	save JksControl md
end 
if any(steps==4)  %GPU solver

	load JksControl  %comment this out when increasing the spatial resolution
        % md = refine(md); Add this when increasing spatial resolution
	addpath('../../src/');
	%md=gpu(md,1.5,.9);
	gpu_parallelized  %running it as a script to obtain all the script scalars in the workspace
end
