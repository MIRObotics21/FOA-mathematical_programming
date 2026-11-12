clear;clc;
load prob2.mat;
n = 2; %dimenze
m = 2; %pocet omezeni
x0 = [0;0]; %vhodne zvolit pocatek aby splnoval omezeni

%spravne udelat funkce, grad a hess
f = @(x) x'*P1*x + c'*x;
f_grad = @(x) 2*P1*x + c;
f_hess = @(x) 2*eye(n);
g1 = @(x) x'*P2*x - 1;
g2 = @(x) exp(x(1))-x(2)-2;
g1_grad = @(x) 2*P2*x;
g2_grad = @(x) [exp(x(1)); -1];
g1_hess = @(x) 2*P2;
g2_hess = @(x) [exp(x(1)), 0; 0, 0];

%zkopcene od Vas
Phi = @(x) -(log(-g1(x)) + log(-g2(x)));
Phi_grad = @(x) - (1/g1(x))*g1_grad(x) - (1/g2(x))*g2_grad(x);
Phi_hess = @(x) (1/g1(x)^2)*g1_grad(x)*(g1_grad(x))' - 1/g1(x)*g1_hess(x)+...
                (1/g2(x)^2)*g2_grad(x)*(g2_grad(x))' - 1/g2(x)*g2_hess(x);

xs = [x0]; t = 0.1; beta = 20; 
vareps = 1e-4;
vareps_inner = 1e-4; 
opt_alphas = cell(1);
iter_out = 1; iters_in = 0;
tic
while m/t > vareps
    t = beta*t;
    obj_f = @(x) t*f(x) + Phi(x);
    obj_grad = @(x) t*f_grad(x) + Phi_grad(x);
    obj_hess = @(x) t*f_hess(x) + Phi_hess(x); 
    g = obj_grad(xs(:,end));
    H = obj_hess(xs(:,end));
    iter_in = 1;
    while 1
        d = -H\g;
        if d'*H*d < vareps_inner
            break
        end
        f_red = @(alpha) obj_f(xs(:,end) + d*alpha);
        [b_left, b_right] = bracket_minimum(f_red);
        [b_left, b_right] = golden_section_search(f_red,b_left, b_right,20);
        alpha = (b_left + b_right)/2;
        opt_alphas{iter_out,1}(iter_in,1) = alpha;
        xs(:,end+1) = xs(:,end) + alpha*d;
        g = obj_grad(xs(:,end));
        H = obj_hess(xs(:,end));
        iter_in = iter_in + 1;
    end
    iters_in(iter_out) = iter_in;
    iter_out = iter_out + 1;
end
toc

x_best = xs(:,end);
fprintf('nejlepsi x found at: [%f %f] \n', x_best(1),x_best(2))
fprintf('s fkcni hodnotou: %d \n', f(x_best))

disp('optional kontrola splneni podminek')
[g1(x_best), g2(x_best)]

% hold on
% grid on
% f1 = @(x,y) [x;y]'*P1*[x;y] -1 ;
% f2 = @(x,y) [x;y]'*P2*[x;y] -1 ;
% fimplicit(f1);
% fimplicit(f2);
% plot(xs(1,:),xs(2,:),'r-x');
% plot(x_0(1),x_0(2),'kx')
