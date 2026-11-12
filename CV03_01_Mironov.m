clear;clc;close all;
f = @(x) rosenbrock(x); range = [-1.5,1.5]; opt = [1,1]; conts = 0:100;
grad = @(x) rosenbrock_grad(x);
hess = @(x) rosenbrock_hessian(x);

x0 = [-1,-1];
max_iter = 100;
max_fcalls = 1000;

%main
%cyclic
epsilon = 1e-4;

x = cyclic(f,x0,epsilon,max_iter,max_fcalls);
figure(1)
hold on;
title('Cyclic coordinate search')
plot_f(f, range, opt, conts);
plot(x(:,1),x(:,2),'r-', 'LineWidth', 1)
hold off

%h-j
alpha = 1;
epsilon = 1e-6;

x = hooke_jeeves(f,x0,alpha,epsilon,0.5,max_iter,max_fcalls);
figure(2)
hold on;
title('Hooke-Jeeves method')
plot_f(f, range, opt, conts);
plot(x(:,1),x(:,2),'r-', 'LineWidth', 1)
hold off

%n-m
epsilon = 1e-6;
S = [-1 -0.8 -0.5; -1 -0.8 -1];

x = nelder_mead(f, S, epsilon, 1.4, 1.7, 1, max_iter, max_fcalls);
figure(3)
hold on;
title('Nelder-Mead method')
plot_f(f, range, opt, conts);
for i = 1:length(x)/3
    idx = (i - 1) * 3 + (1:3);
    x1 = x(1, idx);
    x2 = x(2, idx);
    
    x1 = [x1, x1(1)];
    x2 = [x2, x2(1)];
    
    plot(x1, x2, 'r-', 'LineWidth', 1);
end
hold off

%q-n
epsilon = 1e-2;

x = quasi_newton(f, grad, x0, epsilon, max_iter, max_fcalls)';
figure(4)
hold on;
title('Quasi-newton method')
plot_f(f, range, opt, conts);
plot(x(1,:),x(2,:),'r-', 'LineWidth', 1)
hold off

%%
%plotovani rosenbrocka
function plot_f(f, range, opt, conts)
    xs = linspace(range(1),range(2));
    ys = linspace(range(1),range(2));
    [Xs,Ys] = meshgrid(xs,ys);
    for i=1:100
        for j=1:100
            Zs(i,j) = f([Xs(i,j),Ys(i,j)]);
        end
    end
    contour(Xs,Ys,Zs,conts); hold on; grid on; axis equal;
    plot3(opt(1),opt(2),0,'ko'); 
end

%funkce rosenbrocka
function [val] = rosenbrock(x)
    a = 1; b = 5;
    val = (a-x(1))^2 + b*(x(2)-x(1)^2)^2;
end

%gradient rosenbrocka
function [val] = rosenbrock_grad(x)
    a = 1; b = 5;
    val = [-2*(a-x(1))+2*b*(x(2)-x(1)^2)*(-2*x(1)); 2*b*(x(2)-x(1)^2)];
end

%gsearch se vstupem funkce, start, smer
function [optimum] = golden_search(f, x, direction)
    max_iter = 20;
    range = 2;
    phi = (1 + sqrt(5)) / 2;

    direction = direction(:)';
    direction = direction / norm(direction);

    a = 0;
    b = range;

    lambda1 = b - (b - a) / phi;
    lambda2 = a + (b - a) / phi;

    x1 = x + lambda1 * direction;
    x2 = x + lambda2 * direction;

    f1 = f(x1);
    f2 = f(x2);

    for iter = 1:max_iter
        if f1 > f2
            a = lambda1;
            lambda1 = lambda2;
            lambda2 = a + (b - a) / phi;

            x1 = x2;
            f1 = f2;

            x2 = x + lambda2 * direction;
            f2 = f(x2);
        else
            b = lambda2;
            lambda2 = lambda1;
            lambda1 = b - (b - a) / phi;

            x2 = x1;
            f2 = f1;

            x1 = x + lambda1 * direction;
            f1 = f(x1);
        end
    end
    lambda_opt = (a + b) / 2;
    optimum = x + lambda_opt * direction;
end

%cyclic
function [x_history] = cyclic(f, x, epsilon, max_iter, max_fcalls)
    delta = inf;
    n = length(x);
    x_history = [x];
    iter = 0;
    fcalls = 0;
    while abs(delta) > epsilon && iter < max_iter && fcalls < max_fcalls
        x1 = x;
        for i=1:n
            if fcalls >= max_fcalls
                break
            end
            d = zeros(n,1);
            d(i) = 1;
            x = golden_search(f, x, d);
            fcalls = fcalls+4;
            x_history = [x_history; x];
        end
        if fcalls >= max_fcalls
            break
        end
        x = golden_search(f, x, (x-x1));
        fcalls = fcalls+4;
        x_history = [x_history; x];
        delta = norm(x-x1);
        iter = iter+1;
    end
    fprintf('cyclic: \n iterations = %d \n fcalls = %d \n', iter, fcalls)
end

%h-j
function [x_history] = hooke_jeeves(f, x, alpha, epsilon, gamma, max_iter, max_fcalls)
    y = f(x);
    n = length(x);
    x_history = [x];
    iter = 0;
    fcalls = 1;
    while alpha > epsilon && iter < max_iter && fcalls < max_fcalls
        improved = 0;
        x_best = x;
        y_best = y;
        for i=1:n
            for sgn=[-1,1]
                d = zeros(n,1);
                d(i) = sgn;
                x_temp = x+sgn*alpha*d';
                y_temp = f(x_temp);
                fcalls = fcalls+1;
                if y_temp < y_best
                    x_best = x_temp;
                    y_best = y_temp;
                    improved = 1;
                    x_history = [x_history; x_best];
                end
            end
        end
        x = x_best;
        y = y_best;
        if improved == 0
            alpha = alpha*gamma;
        end
        iter = iter+1;
    end
    fprintf('hooke-jeeves: \n iterations = %d \n fcalls = %d \n', iter, fcalls)
end

%n-m
function [x_history] = nelder_mead(f, S, epsilon, alpha, beta, gamma, max_iter, max_fcalls)
    delta = inf;
    iter = 0;
    fcalls = 0;
    y = [f(S(:,1)) f(S(:,2)) f(S(:,3))];
    fcalls = fcalls +3;
    x_history = [];
    while abs(delta) > epsilon && iter < max_iter && fcalls < max_fcalls
        [y, p] = sort(y);
        S = S(:,p);
        x_history = [x_history,S];
        xl = S(:,1);
        yl = y(:,1);
        xh = S(:,end);
        yh = y(:,end);
        xs = S(:,end-1);
        ys = y(:,end-1);
        xm = [(xl(1)+xh(1)+xs(1))/3;(xl(2)+xh(2)+xs(2))/3];
        xr = xm + alpha*(xm-xh);
        yr = f(xr);
        fcalls = fcalls + 1;
        if yr < yl
            xe = xm + beta*(xr-xm);
            ye = f(xe);
            fcalls = fcalls + 1;
            if ye < yr
                S(:,end) = xe;
                y(:,end) = ye;
            else
                S(:,end) = xr;
                y(:,end) = yr;
            end
        elseif yr > ys
            if yr <= yh
                xh = xr;
                yh = yr;
                S(:,end) = xr;
                y(:,end) = yr;
            end
            xc = xm + gamma*(xh-xm);
            yc = f(xc);
            fcalls = fcalls + 1;
            if yc > yh
                for i=2:length(yh)
                    S(:,i) = (S(:,i)+xl)/2;
                    y(:,i) = f(S(:,i));
                end
            else
                S(:,end) = xc;
                y(:,end) = yr;
            end
        else
            S(:,end) = xr;
            y(:,end) = yr;
        end
        delta = std(y);
        iter = iter+1;
    end
    fprintf('nelder-mead: \n iterations = %d \n fcalls = %d \n', iter, fcalls)
end

%q-n
function [x_history] = quasi_newton(f, grad, x, epsilon, max_iter, max_fcalls)
    fcalls = 0;
    iter = 0;
    n = length(x);
    Q = eye(n); % Initial Hessian approximation (Identity matrix)
    x_history = [x];
    while norm(grad(x)) > epsilon && iter < max_iter && fcalls < max_fcalls
        g = grad(x);
        fcalls = fcalls + 1;
        x_new = golden_search(f,x,-Q*g);
        fcalls = fcalls + 4;
        g_new = grad(x_new);
        fcalls = fcalls + 1;
        delta = x_new-x;
        gamma = g_new-g;
        Q = Q - (Q*(gamma*gamma')*Q)/(gamma'*Q*gamma)+(delta*delta')/(delta*gamma);
        x = x_new;
        x_history = [x_history;x];
        iter = iter + 1;
    end
    fprintf('quasi-newton: \n iterations = %d \n fcalls = %d \n', iter, fcalls)
end