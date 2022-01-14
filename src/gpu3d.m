function md = gpu3d(md,damp,relaxation)
%GPU - solve HO stress balance using a GPU-like algorithm
%
%   Usage:
%      md = gpu(md,damp,relaxation)
%
%   Defaults:
%      damp  = 0.2;   NOTE: 0<damp<20
%      relax = 1;     NOTE: 1 = no relaxation, <1 = under-relaxation (more stable)

%Set default arguments
if nargin<1
	error('not enough input arguments');
elseif nargin<2
	relaxation = 1;
	damp       = 0.2;
elseif nargin<3
	relaxation = 1;
end

% ======================== LOAD DATA FROM MD ===============================

nbe              = md.mesh.numberofelements;
nbv              = md.mesh.numberofvertices;
g                = md.constants.g;
rho              = md.materials.rho_ice;
rho_w            = md.materials.rho_water;
yts              = md.constants.yts;
index            = md.mesh.elements;
spcvx            = md.stressbalance.spcvx/yts;
spcvy            = md.stressbalance.spcvy/yts;
x                = md.mesh.x;
y                = md.mesh.y;
z                = md.mesh.z;
H                = md.geometry.thickness;
surface          = md.geometry.surface;
base             = md.geometry.base;
ice_levelset     = md.mask.ice_levelset;
ocean_levelset   = md.mask.ocean_levelset;
rheology_B_temp  = md.materials.rheology_B;
vx               = md.initialization.vx/yts;
vy               = md.initialization.vy/yts;
friction         = md.friction.coefficient;

% =================== SIMILAR TO C++ FILE LINE 384 DOWN ===================

%Constants 
n_glen    = 3;
rele      = 1e-1;
eta_b     = 0.5;
eta_0     = 1.e+14/2.;
niter     = 5e6;
nout_iter = 1000;
epsi      = 1e-8;

%Initial guesses (except vx and vy that we already loaded)
etan   = 1.e+14*ones(nbe,1);
dVxdt  = zeros(nbv,1);
dVydt  = zeros(nbv,1);

%Manage derivatives once for all
%[areas alpha beta gamma]=NodalCoeffs(index,x,y,nbe);
%weights=Weights(index,areas,nbe,nbv);

%Mesh size
%[resolx resoly] = MeshSize(index,x,y,areas,weights,nbe,nbv);

%Physical properties once for all
%[dsdx dsdy] = derive_xy_elem(surface,index,alpha,beta,nbe);
Helem       = mean(H(index),2);
rheology_B  = mean(rheology_B_temp(index),2);


%Linear integration points (orders 3 and 5)
lin3_wgt = [0.555555555555556, 0.888888888888889, 0.555555555555556];
lin3_xg  = [-0.774596669241483, 0.000000000000000, 0.774596669241483];
lin5_wgt = [0.5688888888888889,0.4786286704993665,0.4786286704993665,0.2369268850561891,0.2369268850561891];
lin5_xg  = [0.0000000000000000,-0.5384693101056831,0.5384693101056831,-0.9061798459386640,0.9061798459386640];
%Tria integration points (orders 3 and 5)
tria3_wgt  = [-0.974278579257493, 0.902109795608790, 0.902109795608790, 0.902109795608790];
tria3_xg_1 = [0.333333333333333, 0.600000000000000, 0.200000000000000, 0.200000000000000];
tria3_xg_2 = [0.333333333333333, 0.200000000000000, 0.600000000000000, 0.200000000000000];
tria3_xg_3 = [0.333333333333333, 0.200000000000000, 0.200000000000000, 0.600000000000000];
tria5_wgt  = [0.389711431702997, 0.229313399254729, 0.229313399254729, 0.229313399254729, 0.218133059367230, 0.218133059367230, 0.218133059367230];
tria5_xg_1 = [0.333333333333333, 0.059715871789770, 0.470142064105115, 0.470142064105115, 0.797426985353087, 0.101286507323456, 0.101286507323456];
tria5_xg_2 = [0.333333333333333, 0.470142064105115, 0.059715871789770, 0.470142064105115, 0.101286507323456, 0.797426985353087, 0.101286507323456];
tria5_xg_3 = [0.333333333333333, 0.470142064105115, 0.470142064105115, 0.059715871789770, 0.101286507323456, 0.101286507323456, 0.797426985353087];
%Penta integration points (orders 3 and 5)
numg3 = 4*3;
penta3_wgt  = zeros(numg3,1);
penta3_xg_1 = zeros(numg3,1);
penta3_xg_2 = zeros(numg3,1);
penta3_xg_3 = zeros(numg3,1);
penta3_xg_4 = zeros(numg3,1);
for i=1:4
   for j=1:3
      ig = 3*(j-1)+i;
      penta3_wgt(ig)  = tria3_wgt(i)*lin3_wgt(j);
      penta3_xg_1(ig) = tria3_xg_1(i);
      penta3_xg_2(ig) = tria3_xg_2(i);
      penta3_xg_3(ig) = tria3_xg_3(i);
      penta3_xg_4(ig) = lin3_xg(j);
   end
end
numg5 = 7*5;
penta5_wgt  = zeros(numg5,1);
penta5_xg_1 = zeros(numg5,1);
penta5_xg_2 = zeros(numg5,1);
penta5_xg_3 = zeros(numg5,1);
penta5_xg_4 = zeros(numg5,1);
for i=1:7
   for j=1:5
      ig = 5*(j-1)+i;
      penta5_wgt(ig)  = tria5_wgt(i)*lin5_wgt(j);
      penta5_xg_1(ig) = tria5_xg_1(i);
      penta5_xg_2(ig) = tria5_xg_2(i);
      penta5_xg_3(ig) = tria5_xg_3(i);
      penta5_xg_4(ig) = lin5_xg(j);
   end
end

%Compute RHS amd ML once for all
ML            = zeros(nbv,1);
Fvx           = zeros(nbv,1);
Fvy           = zeros(nbv,1);
groundedratio = zeros(nbe,1);
isice         = false(nbe,1);
for n=1:nbe

   for ig=1:numg3
      %Compute basis functions
      basis = zeros(6,1);
      basis(1) = penta5_xg_1*(1.-penta5_xg_4)/2.;
      basis(2) = penta5_xg_2*(1.-penta5_xg_4)/2.;
      basis(3) = penta5_xg_3*(1.-penta5_xg_4)/2.;
      basis(4) = penta5_xg_1*(1.+penta5_xg_4)/2.;
      basis(5) = penta5_xg_2*(1.+penta5_xg_4)/2.;
      basis(6) = penta5_xg_3*(1.+penta5_xg_4)/2.;

		%Get Jacobian determinant (TODO)
		Jdet = XXX;

      %Lumped mass matrix
      for i=1:6
         for j=1:6
				ML(index(n,j)) = ML(index(n,j)) + penta5_wgt(ig)*Jdet*basis(i)*basis(j);
         end
      end
   end

	%Is there ice at all in the current element?
	level = ice_levelset(index(n,:));
	if level(1)<0 || level(2)<0 || level(3)<0
		isice(n)= true;
	else
		%We can skip this element altogether
		isice(n) = false;
		for i=1:6
			vx(index(n,i)) = 0.;
			vy(index(n,i)) = 0.;
		end
		continue;
	end

	%RHS (Driving stress term) Later....
   %RHS (Water pressure at the ice front) Later ...
end

%Finally add calculation of friction coefficient:
alpha2 = zeros(nbv,1);
for i=1:nbv

   %Compute effective pressure
   p_ice   = g*rho*H(i);
   p_water = -rho_w*g*base(i);
   Neff    = p_ice - p_water;
   if(Neff<0.) Neff=0.; end

   %Compute alpha2
   alpha2(i) = friction(i)^2*Neff;
end

%Update viscosity
[dvxdx dvxdy]=derive_xy_elem(vx,index,alpha,beta,nbe);
[dvydx dvydy]=derive_xy_elem(vy,index,alpha,beta,nbe);

for i=1:nbe
if ~isice(i) continue; end
eps_xx = dvxdx(i);
eps_yy = dvydy(i);
eps_xy = .5*(dvxdy(i)+dvydx(i));
EII2 = eps_xx^2 + eps_yy^2 + eps_xy^2 + eps_xx*eps_yy;
eta_it = 1.e+14/2.;
if(EII2>0.) eta_it = rheology_B(i)/(2*EII2^((n_glen-1.)/(2*n_glen))); end

etan(i) = min(eta_it,eta_0*1e5);
if(isnan(etan(i))) error('Found NaN in etan(i)'); end
end


%Main loop, allocate a few vectors needed for the computation

tic
for iter = 1:niter % Pseudo-Transient cycles

	%Strain rates
	[dvxdx dvxdy]=derive_xy_elem(vx,index,alpha,beta,nbe);
	[dvydx dvydy]=derive_xy_elem(vy,index,alpha,beta,nbe);

	%KV term in equation 22
	KVx  = zeros(nbv,1);
	KVy  = zeros(nbv,1);
	for n=1:nbe

		%Skip if no ice
		if ~isice(n) continue; end

		%Viscous Deformation
		eta_e  = etan(n);
		eps_xx = dvxdx(n);
		eps_yy = dvydy(n);
		eps_xy = .5*(dvxdy(n)+dvydx(n));
		for i=1:3
			KVx(index(n,i)) = KVx(index(n,i))+ ...
				2*Helem(n)*eta_e*(2*eps_xx+eps_yy)*alpha(n,i)*areas(n) + ...
				2*Helem(n)*eta_e*eps_xy*beta(n,i)*areas(n);
			KVy(index(n,i)) = KVy(index(n,i))+ ...
				2*Helem(n)*eta_e*eps_xy*alpha(n,i)*areas(n) + ...
				2*Helem(n)*eta_e*(2*eps_yy+eps_xx)*beta(n,i)*areas(n);
		end
		%Add basal friction
		if groundedratio(n)>0.
         for k=1:3
            for i=1:3
               for j=1:3
                  if i==j && j==k
                     KVx(index(n,k)) = KVx(index(n,k))+ groundedratio(n)*alpha2(index(n,i))*vx(index(n,j))*areas(n)/10.;
                     KVy(index(n,k)) = KVy(index(n,k))+ groundedratio(n)*alpha2(index(n,i))*vy(index(n,j))*areas(n)/10.;
                  elseif (i~=j) && (j~=k) && (k~=i)
                     KVx(index(n,k)) = KVx(index(n,k))+ groundedratio(n)*alpha2(index(n,i))*vx(index(n,j))*areas(n)/60.;
                     KVy(index(n,k)) = KVy(index(n,k))+ groundedratio(n)*alpha2(index(n,i))*vy(index(n,j))*areas(n)/60.;
                  else
                     KVx(index(n,k)) = KVx(index(n,k))+ groundedratio(n)*alpha2(index(n,i))*vx(index(n,j))*areas(n)/30.;
                     KVy(index(n,k)) = KVy(index(n,k))+ groundedratio(n)*alpha2(index(n,i))*vy(index(n,j))*areas(n)/30.;
                  end
               end
            end
         end
		end
	end

	%Get current viscosity on nodes (Needed for time stepping)
	eta_nbv = elem2node(etan,index,areas,weights,nbe,nbv);

	%Velocity rate update in the x and y, refer to equation 19 in Rass paper
	normX = 0;
	normY = 0;

	%1. Get time derivative based on residual (dV/dt)
	ResVx =  1./(rho*ML).*(-KVx + Fvx); %rate of velocity in the x, equation 23
	ResVy =  1./(rho*ML).*(-KVy + Fvy); %rate of velocity in the y, equation 24
	dVxdt = dVxdt*(1.-damp/20.) + ResVx;
	dVydt = dVydt*(1.-damp/20.) + ResVy;
	if(any(isnan(dVxdt))) error('Found NaN in dVxdt[i]'); end
	if(any(isnan(dVydt))) error('Found NaN in dVydt[i]'); end

	%2. Explicit CFL time step for viscous flow, x and y directions
	dtVx = rho*resolx.^2./(4*max(80,H).*eta_nbv*(1.+eta_b)*4.1);
	dtVy = rho*resoly.^2./(4*max(80,H).*eta_nbv*(1.+eta_b)*4.1);
        
	%3. velocity update, vx(new) = vx(old) + change in vx, Similarly for vy
	vx = vx + relaxation*dVxdt.*dtVx;
	vy = vy + relaxation*dVydt.*dtVy;

	%Apply Dirichlet boundary condition, Residual should also be 0 (for convergence)
	pos = find(~isnan(spcvx));
	vx(pos) = spcvx(pos);
	dVxdt(pos) = 0.;
	pos = find(~isnan(spcvy));
	vy(pos) = spcvy(pos);
	dVydt(pos) = 0.;

	%4. Update error
	normX = normX + sum(dVxdt.^2);
	normY = normY + sum(dVydt.^2);

	%Get final error estimate
	normX = sqrt(normX)/nbv;
	normY = sqrt(normY)/nbv;

	%Check convergence
	iterror = max(normX,normY);
	if((iterror < epsi) && (iter > 2)) break; end
	if (mod(iter,nout_iter)==1)
		fprintf('iter=%d, err=%1.3e \n',iter,iterror)
		clf
		plotmodel(md,'data',sqrt(vx.^2+vy.^2)*yts);
		drawnow
	end

	%LAST: Update viscosity
	for i=1:nbe
		if ~isice(i) continue; end
		eps_xx = dvxdx(i);
		eps_yy = dvydy(i);
		eps_xy = .5*(dvxdy(i)+dvydx(i));
		EII2 = eps_xx^2 + eps_yy^2 + eps_xy^2 + eps_xx*eps_yy;
		eta_it = 1.e+14/2.;
		if(EII2>0.) eta_it = rheology_B(i)/(2*EII2^((n_glen-1.)/(2*n_glen))); end

		etan(i) = min(exp(rele*log(eta_it) + (1-rele)*log(etan(i))),eta_0*1e5);
		if(isnan(etan(i))) error('Found NaN in etan(i)'); end
	end
end
clf
plotmodel(md,'data',sqrt(vx.^2+vy.^2)*yts);
fprintf('iter=%d, err=%1.3e --> converged\n',iter,iterror)
toc
end

function [areas alpha beta gamma]=NodalCoeffs(index,x,y,nbe)% {{{
	[alpha beta gamma]=GetNodalFunctionsCoeff(index,x,y);
	areas=GetAreas(index,x,y);
end % }}}
function weights=Weights(index,areas,nbe,nbv)% {{{
	weights = sparse(index(:),ones(3*nbe,1),repmat(areas,3,1),nbv,1);
end % }}}
function [dfdx_e dfdy_e] = derive_xy_elem(f,index,alpha,beta,nbe)% {{{

	dfdx_e = sum(f(index).*alpha,2);
	dfdy_e = sum(f(index).*beta,2);

end % }}}
function f_v = elem2node(f_e,index,areas,weights,nbe,nbv)% {{{

	f_v = sparse(index(:),ones(3*nbe,1),repmat(areas.*f_e,3,1),nbv,1)./weights;

end % }}}
function [resolx resoly] = MeshSize(index,x,y,areas,weights,nbe,nbv)% {{{

	%Get element size
	dx_elem=max(x(index),[],2)-min(x(index),[],2);
	dy_elem=max(y(index),[],2)-min(y(index),[],2);

	%Average over each node
	resolx = elem2node(dx_elem,index,areas,weights,nbe,nbv);
	resoly = elem2node(dy_elem,index,areas,weights,nbe,nbv);
end % }}}
