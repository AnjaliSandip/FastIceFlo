%md=squaremesh(model(),1000000,1000000,20,20);
md=triangle(model(),'./TestFiles/Square.exp',50000.);
%md=triangle(model(),'./TestFiles/Square.exp',20000.);
md=setmask(md,'all','');
md=parameterize(md,'./TestFiles/SquareShelfConstrained.par');
md.stressbalance.restol=1e-6;
md=setflowequation(md,'SSA','all');

%Linear case
%md.materials.rheology_n(:)=1;
%md.materials.rheology_B(:)=1e14*2;
%md=solve(md,'sb');

% ======================== LOAD DATA FROM MD ===============================

nbe              = md.mesh.numberofelements;
nbv              = md.mesh.numberofvertices;
g                = md.constants.g;
rho              = md.materials.rho_ice;
yts              = md.constants.yts;
index            = md.mesh.elements;
vertexonboundary = md.mesh.vertexonboundary;
x                = md.mesh.x;
y                = md.mesh.y;
H                = md.geometry.thickness;
surface          = md.geometry.surface;
rheology_B_temp  = md.materials.rheology_B;
vx               = md.initialization.vx/yts;
vy               = md.initialization.vy/yts;

% =================== SIMILAR TO C++ FILE LINE 384 DOWN ===================

%Constants 
n_glen    = 3;
damp      = 2;
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
[areas alpha beta gamma]=NodalCoeffs(index,x,y,nbe);
weights=Weights(index,areas,nbe,nbv);

%Mesh size
[resolx resoly] = MeshSize(index,x,y,areas,weights,nbe,nbv);

%Physical properties once for all
[dsdx dsdy] = derive_xy_elem(surface,index,alpha,beta,nbe);
Helem       = mean(H(index),2);
rheology_B  = mean(rheology_B_temp(index),2);

%Compute RHS amd ML once for all
ML  = zeros(nbv,1);
Fvx = zeros(nbv,1);
Fvy = zeros(nbv,1);
for n=1:nbe
	%Lumped mass matrix
	for i=1:3
		for j=1:3
			% \int_E phi_i * phi_i dE = A/6 and % \int_E phi_i * phi_j dE = A/12
			if(i==j)
				ML(index(n,j)) = ML(index(n,j))+areas(n)/6;
			else
				ML(index(n,j)) = ML(index(n,j))+areas(n)/12;
			end
		end
	end
	%RHS
	for i=1:3
		for j=1:3
			if i==j
				Fvx(index(n,i)) = Fvx(index(n,i)) -rho*g*H(index(n,j))*dsdx(n)*areas(n)/6.;
				Fvy(index(n,i)) = Fvy(index(n,i)) -rho*g*H(index(n,j))*dsdy(n)*areas(n)/6.;
			else
				Fvx(index(n,i)) = Fvx(index(n,i)) -rho*g*H(index(n,j))*dsdx(n)*areas(n)/12.;
				Fvy(index(n,i)) = Fvy(index(n,i)) -rho*g*H(index(n,j))*dsdy(n)*areas(n)/12.;
			end
		end
	end
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
	end

	%Get current viscosity on nodes (Needed for time stepping)
	eta_nbv = elem2node(etan,index,areas,weights,nbe,nbv);

	%Velocity rate update in the x and y, refer to equation 19 in Rass paper
	normX = 0;
	normY = 0;
	for i=1:nbv

		%1. Get time derivative based on residual (dV/dt)
		ResVx =  1./(rho*ML(i))*(-KVx(i) + Fvx(i)); %rate of velocity in the x, equation 23
		ResVy =  1./(rho*ML(i))*(-KVy(i) + Fvy(i)); %rate of velocity in the y, equation 24
		dVxdt(i) = dVxdt(i)*(1.-damp/20.) + ResVx;
		dVydt(i) = dVydt(i)*(1.-damp/20.) + ResVy;
		if(isnan(dVxdt(i))) error('Found NaN in dVxdt[i]'); end
		if(isnan(dVydt(i))) error('Found NaN in dVydt[i]'); end

		%2. Explicit CFL time step for viscous flow, x and y directions
		dtVx = rho*resolx(i)^2/(4*H(i)*eta_nbv(i)*(1.+eta_b)*4.1);
		dtVy = rho*resoly(i)^2/(4*H(i)*eta_nbv(i)*(1.+eta_b)*4.1);

		%3. velocity update, vx(new) = vx(old) + change in vx, Similarly for vy
		vx(i) = vx(i) + dVxdt(i)*dtVx;
		vy(i) = vy(i) + dVydt(i)*dtVy;

		%Apply Dirichlet boundary condition
		if(vertexonboundary(i))
			vx(i) = 0.;
			vy(i) = 0.;

			%Residual should also be 0 (for convergence)
			dVxdt(i) = 0.;
			dVydt(i) = 0.;
		end

		%4. Update error
		normX = normX + dVxdt(i)^2;
		normY = normY + dVydt(i)^2;
	end

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
toc
clf
plotmodel(md,'data',sqrt(vx.^2+vy.^2)*yts);
fprintf('iter=%d, err=%1.3e \n',iter,iterror)

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
