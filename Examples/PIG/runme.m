steps=[6];

if any(steps==1) %Mesh Generation #1

	%Mesh parameters
	domain =['./DomainOutline.exp'];
	hinit=5000;   % element size for the initial mesh
	hmax=40000;    % maximum element size of the final mesh
	hmin=4000;     % minimum element size of the final mesh
	gradation=1.7; % maximum size ratio between two neighboring elements
	err=8;         % maximum error between interpolated and control field

	% Generate an initial uniform mesh (resolution = hinit m)
	md=bamg(model,'domain',domain,'hmax',hinit);

	% Get necessary data to build up the velocity grid
	nsidc_vel= [issmdir() 'examples/Data/Antarctica_ice_velocity.nc'];
	xmin    = strsplit(ncreadatt(nsidc_vel,'/','xmin'));      xmin    = str2num(xmin{2});
	ymax    = strsplit(ncreadatt(nsidc_vel,'/','ymax'));      ymax    = str2num(ymax{2});
	spacing = strsplit(ncreadatt(nsidc_vel,'/','spacing'));   spacing = str2num(spacing{2});
	nx      = double(ncreadatt(nsidc_vel,'/','nx'));
	ny      = double(ncreadatt(nsidc_vel,'/','ny'));
	vx      = double(ncread(nsidc_vel,'vx'));
	vy      = double(ncread(nsidc_vel,'vy'));

	% Build the coordinates
	x=xmin+(0:1:nx)'*spacing;
	y=(ymax-ny*spacing)+(0:1:ny)'*spacing;
	
	% Interpolate velocities onto coarse mesh
	vx_obs=InterpFromGridToMesh(x,y,flipud(vx'),md.mesh.x,md.mesh.y,0);
	vy_obs=InterpFromGridToMesh(x,y,flipud(vy'),md.mesh.x,md.mesh.y,0);
	vel_obs=sqrt(vx_obs.^2+vy_obs.^2);
	clear vx vy x y;

	% Adapt the mesh to minimize error in velocity interpolation
	md=bamg(md,'hmax',hmax,'hmin',hmin,'gradation',gradation,'field',vel_obs,'err',err);

	%save model
	save ./Models/PIG_Mesh_generation md;
end

if any(steps==2)  %Masks #2

	md = loadmodel('./Models/PIG_Mesh_generation');	

	disp('   -- Interpolating from BedMachine');
	md.mask.ice_levelset				= -1*ones(md.mesh.numberofvertices,1); % set 'presence of ice' everywhere
	md.mask.ocean_levelset			= +1*ones(md.mesh.numberofvertices,1); % set 'grounded ice' everywhere
	mask= interpBedmachineAntarctica(md.mesh.x,md.mesh.y,'mask','nearest',[issmdir() 'examples/Data/BedMachineAntarctica_2020-07-15_v02.nc']); % interp method: nearest
	pos = find(mask<1); % we also want a bit of ice where there are rocks, so keeping ice where mask==1
	md.mask.ice_levelset(pos)	= 1; % set 'no ice' only in the ocean part
	pos = find(mask==0 | mask==3); 
	md.mask.ocean_levelset(pos)=-1; % set 'floating ice' on the ocean part and on the ice shelves

	save ./Models/PIG_SetMask md;
end

if any(steps==3)  %Parameterization #3

	md = loadmodel('./Models/PIG_SetMask');
	md = setflowequation(md,'SSA','all');
	md = parameterize(md,'./Pig.par');
	
	save ./Models/PIG_Parameterization md;
end

if any(steps==4)  %Rheology B inversion

	md = loadmodel('./Models/PIG_Parameterization');

	% Control general
	md.inversion.iscontrol=1;
	md.inversion.maxsteps=40;
	md.inversion.maxiter=40;
	md.inversion.dxmin=0.1;
	md.inversion.gttol=1.0e-6;
	md.verbose=verbose('control',true);

	% Cost functions
	md.inversion.cost_functions=[101 103 502];
	md.inversion.cost_functions_coefficients=ones(md.mesh.numberofvertices,3);
	md.inversion.cost_functions_coefficients(:,1)=1000;
	md.inversion.cost_functions_coefficients(:,2)=1;
	md.inversion.cost_functions_coefficients(:,3)=1.e-16;

	% Controls
	md.inversion.control_parameters={'MaterialsRheologyBbar'};
	md.inversion.min_parameters=md.materials.rheology_B;
	md.inversion.max_parameters=md.materials.rheology_B;
	pos = find(md.mask.ocean_levelset<0);
	md.inversion.min_parameters(pos) = cuffey(273);
	md.inversion.max_parameters(pos) = cuffey(200);

	% Additional parameters
	md.stressbalance.restol=0.01;
	md.stressbalance.reltol=0.1;
	md.stressbalance.abstol=NaN;

	% Solve
	md.cluster=generic('name',oshostname,'np',2);
	mds=extract(md,md.mask.ocean_levelset<0);
	mds=solve(mds,'Stressbalance');

	% Update model rheology_B accordingly
	md.materials.rheology_B(mds.mesh.extractedvertices)=mds.results.StressbalanceSolution.MaterialsRheologyBbar;

	% Save model
	save ./Models/PIG_Control_B md;
end

if any(steps==5)  %drag inversion

	md = loadmodel('./Models/PIG_Control_B');

	% Cost functions
	md.inversion.cost_functions=[101 103 501];
	md.inversion.cost_functions_coefficients=ones(md.mesh.numberofvertices,3);
	md.inversion.cost_functions_coefficients(:,1)=2000;
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

	save ./Models/PIG_Control_drag md;
end

if any(steps==6)  %GPU solver

	load ./Models/PIG_Control_drag

	addpath('../../src/');
	md=gpu(md,0.35,0.28);
end
