clear
print_fig = 1; % print paper Figure 1
%% symbolic functions and variables
syms V(tau,x) tauxx(tau,x)
syms Lx theta_r rho alpha eta H Re Da V0 k positive
syms lambda_k
%% governing equations
eq1         =   rho*H*diff(V(tau,x),tau)     + diff(tauxx(tau,x),x) + alpha^2*V(tau,x);  % momentum balance
eq2         = theta_r*diff(tauxx(tau,x),tau) + tauxx(tau,x) + 4*H*eta*diff(V(tau,x),x);  % viscous stress
%% equation for H
eq_V        = expand(diff(eq1,tau)*theta_r - diff(eq2,x) + eq1);
%% scales and nondimensional variables
V_p         = sqrt(4*H*eta/(rho*H)/theta_r);                                             % velocity scale - wave velocity
rho         = solve(Re == (rho*H)*V_p*Lx/(4*H*eta),rho);                                 % density from Reynolds number Re
alpha       = solve(Da == Lx^2*alpha^2/(4*H*eta),alpha);
%% dispersion relation
V(tau,x)    = V0*exp(-lambda_k*V_p*tau/Lx)*sin(pi*k*x/Lx);                        % Fourier expansion term
disp_rel    = expand(subs(subs(eq_V/V(tau,x))));
cfs         = coeffs(disp_rel,lambda_k);
disp_rel    = collect(simplify(disp_rel/cfs(end)),[lambda_k,pi,k]);
cfs         = coeffs(disp_rel,lambda_k);
%% optimal iteration parameters
a           = cfs(3);
b           = cfs(2);
c           = cfs(1);
discrim     = b^2 - 4*a*c;                                                        % quadratic polynomial discriminant
Re_opt      = solve(subs(discrim,k,1),Re);Re_opt=Re_opt(1);
%% evaluate the solution numerically
fun_cfs     = matlabFunction(fliplr(subs(cfs,k,1)));
fun_Re_opt  = matlabFunction(Re_opt);
Da1         = linspace(1e-6,100,601);                                             % create 1D grid of Da values
Re1         = linspace(pi/2,5*pi,601);                                            % create 1D grid of Re values
[Re2,Da2]   = ndgrid(Re1,Da1);                                                    % create 2D grid of Re and Da values
num_lam     = arrayfun(@(Re,Da)(min(real(roots(fun_cfs(Da,Re))))),Re2,Da2);       % compute minimum of real part of roots
num_Re_opt  = fun_Re_opt(Da1);
num_lam_opt = arrayfun(@(Re,Da)(min(real(roots(fun_cfs(Da,Re))))),num_Re_opt,Da1);
%% plot the spectral abscissa
figure(1);clf;colormap cool
contourf(Re2,Da2,num_lam,10,'LineWidth',1);cb=colorbar;
% num_iter = 1./num_lam;
% contourf(Re2,Da2,num_iter,logspace(log10(0.05),log10(0.7),30),'LineWidth',0.5);axis square;cb=colorbar;%caxis([0.7 1.9])
hold on
plot(num_Re_opt,Da1,'w--','LineWidth',4)
hold off
ax = gca;
xlabel('$Re$','Interpreter','latex')
ylabel('$Da$','Interpreter','latex')
xticks([pi/2 pi 2*pi 3*pi 4*pi 5*pi])
xticklabels({'$\pi/2$','$\pi$','$2\pi$','$3\pi$','$4\pi$','$5\pi$'})
cb.Label.Interpreter          = 'latex';
cb.Label.String               = '$\mathrm{min}\{\Re(\lambda_k)\}$';
ax.XAxis.TickLabelInterpreter = 'latex';
ax.YAxis.TickLabelInterpreter = 'latex';
set(gca,'FontSize',14)
cb.Label.FontSize = 16;
title(['$' latex(disp_rel) '$'],'Interpreter','latex','FontSize',16)

if print_fig==1
    figure(2);clf;colormap turbo
    set(gcf,'Units','centimeters','Position',[10 10 12 10],'PaperUnits','centimeters','PaperPosition',[0 0 12 10])
    fs = 12;
    num_iter = 1./num_lam;
    axes('Units','normalized','Position',[0.17 0.21 0.6 0.7]);
    contourf(Re2,Da2,num_iter,logspace(log10(0.09),log10(1.5),25),'LineWidth',0.5);axis square;
%     contourf(Re2,Da2,num_iter,'LineWidth',0.5);axis square;
    cb = colorbar;cb.Position(1) = cb.Position(1)+0.13;cb.Position([2 4])=[0.29 0.55]; cb.Position(3)=0.02;
    cb.Label.String='\bfn_{iter}/nx';
    cb.Label.FontSize=fs;
    set(cb,'Ticks',[0.1 0.2 0.4,0.8,1.5]);
    set(cb,'TickLabels',{'0.1','0.2','0.4','0.8','1.5'});
    set(gca,'ColorScale','log')
    Re_f = matlabFunction(Re_opt(1));
    hold on
    plot(Re_f(Da1),Da1,'w--','LineWidth',1)
    text(12*pi,1500,'\bfRe_{opt}','Rotation',63,'Color','w','FontSize',fs,'FontName','Courier')
    hold off
    ax = gca;
    xl=xlabel('\bfRe','Units','normalized');ylabel('\bfDa')
    xticks([pi/2 2*pi 3*pi 4*pi 5*pi])
    xticklabels({'$\pi/2$','$2\pi$','$3\pi$','$4\pi$','$5\pi$'})
    ax.XAxis.TickLabelInterpreter = 'latex';
    set(gca,'FontSize',fs,'FontName','Courier')
%     print(gcf,'-dpng','fig_niter_optimal.png','-r300')
end
