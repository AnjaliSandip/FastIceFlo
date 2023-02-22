

steps = [1:6];

if any(steps==1) %Mesh Generation #1

	%Mesh parameters
	domain =['./DomainOutline.exp'];
	resol=2750;   % average element size 
	md=triangle(model,'DomainOutline.exp',resol);


	%save model
	save ./PIG_Mesh_generation md;
end

if any(steps==2)  %Masks #2

	md = loadmodel('./PIG_Mesh_generation');	

	disp('   -- Interpolating from BedMachine');
	md.mask.ice_levelset				= -1*ones(md.mesh.numberofvertices,1); % set 'presence of ice' everywhere
	md.mask.ocean_levelset			= +1*ones(md.mesh.numberofvertices,1); % set 'grounded ice' everywhere
	mask= interpBedmachineAntarctica(md.mesh.x,md.mesh.y,'mask','nearest',[issmdir() 'examples/Data/BedMachineAntarctica_2020-07-15_v02.nc']); % interp method: nearest
	pos = find(mask<1); % we also want a bit of ice where there are rocks, so keeping ice where mask==1
	md.mask.ice_levelset(pos)	= 1; % set 'no ice' only in the ocean part
	pos = find(mask==0 | mask==3); 
	md.mask.ocean_levelset(pos)=-1; % set 'floating ice' on the ocean part and on the ice shelves

	save ./PIG_SetMask md;
end

if any(steps==3)  %Parameterization #3

	md = loadmodel('./PIG_SetMask');
	md = setflowequation(md,'SSA','all');
	md = parameterize(md,'./Pig.par');
	   
	save ./PIG_Parameterization md;
end

if any(steps==4)  %Rheology B inversion

	md = loadmodel('./PIG_Parameterization');

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
	%md.stressbalance.restol=0.01;
	%md.stressbalance.reltol=0.1;
	%md.stressbalance.abstol=NaN;
        md.stressbalance.restol=1000;
	md.stressbalance.reltol=NaN;
	md.stressbalance.abstol=10;
        md.settings.solver_residue_threshold = 1.e-3;

	% Solve
	md.cluster=generic('name',oshostname,'np',2);
	mds=extract(md,md.mask.ocean_levelset<0);
	mds=solve(mds,'Stressbalance');

	% Update model rheology_B accordingly
	md.materials.rheology_B(mds.mesh.extractedvertices)=mds.results.StressbalanceSolution.MaterialsRheologyBbar;
   
	% Save model
	save ./PIG_Control_B md;
end

if any(steps==5)  %drag inversion

	md = loadmodel('./PIG_Control_B');

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

	save ./PIG2e4 md;   %saving .mat file
end


if any(steps==6)  %bin file generation
        load PIG2e4
	md.inversion.iscontrol = 0;
        md.stressbalance.abstol = 10;
        md.stressbalance.restol = 1000;
        md.stressbalance.reltol = NaN;
        md.settings.solver_residue_threshold = 1.e-4;
        md.verbose.convergence = 1;
        md=solve(md,'sb','batch','yes');
end
