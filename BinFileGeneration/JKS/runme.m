%steps= [4];
%steps= [1:3];
steps=[1:4];

if any(steps==1)
	disp('	Step 1: Mesh creation'); 
    resol = 600;
	md=triangle(model,'Domain.exp',resol);
    [md.mesh.lat md.mesh.long] = xy2ll(md.mesh.x,md.mesh.y,+1,45,70);
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
    %load JKS2e4
    save JKS8e4
    %md=solve(md,'sb','batch','yes')
	%load JksControl
	
	%Mesh refinement, # of elements increase by 4 fold
    %md = refine(md);
   

     %load JKS3e5
      %load JKS2e4_SSA
      %save JKS
   
%md.inversion.iscontrol=0;
    %solver selection
    %md.toolkits.DefaultAnalysis=bcgslbjacobioptions();
    
	% Solve
	%md=solve(md,'Stressbalance');
    

   %addpath('../../src/');
   %damp = 0.96;
  %relaxation = 0.7;
   %gpu_parallelized_NC
 
end