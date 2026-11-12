clear;clc;clf;

%----------ZADANI----------%

func = @(x) flower(x);
func_grad = @(x) flower_grad(x);

constr = @(x) -x(1)^2 - x(2)^2 + 2;
constr_grad = @(x) [-2*x(1); -2*x(2)];

x0 = [-2;-2];
max_iter = 100;
max_fcalls = 10000;

%----------MAIN----------%

% odkomentujete ten, ktery chcete spustit

% direct method - cyclic search
rho = [0.5, 1, 5, 10];
% cyclic_search_constrained(func,constr,rho,x0,max_iter,max_fcalls)

% stochastic or population method - mads
rho = [10];
% mads_constrained(func,constr,rho,x0,max_iter,max_fcalls)

% BFGS Quasi-Newton
rho = [0.01, 0.5, 1, 4, 8, 16];
% BFGS_qn_constrained(func,func_grad,constr,constr_grad,rho,x0,max_iter,max_fcalls)

% BFGS Quasi-Newton with the Augmented Lagrange + equality constaint
rho = [0.01, 0.5, 1, 4, 8, 16];
% BFGS_qn_lagrange_constrained(func,func_grad,constr,constr_grad,rho,x0,max_iter,max_fcalls)

%% direct method - cyclic search
function [] = cyclic_search_constrained(func,constr,rho,x0,max_iter,max_fcalls)
    global function_calls;
    function_calls = 0;
    total_iter = 0;
    for k=1:length(rho)
        clf;
        %viz kochenderfer 10.30
        f = @(x) func(x) + rho(k)*(constr(x) > 0);
        plot_f(f, [-3,3], [0,0]);
        plot(sqrt(2)*sin(linspace(0,2*pi)),sqrt(2)*cos(linspace(0,2*pi)),'k-');
        title('direct method - cyclic search');
    
        %cyclic prevzat z g-drive
        %splnit zadani (garantovat succes) zde lze jen s extremnim epsilon
        %jinak spadne pod constraint
        epsilon = 10;
        [xs,iter] = cyclic_coordinate_descent(f,x0,epsilon,max_iter,max_fcalls);
        total_iter = total_iter + iter;
        plot(xs(1,:),xs(2,:),'r*-');
    
        pause(1);
        x0 = xs(:,end);
    end
    fprintf('cyclic search took: \n iterations = %d \n fcalls = %d \nwith best found at: [%f %f]\n', total_iter, function_calls, x0(1), x0(2))
end
%%  stochastic or population method - mads
function [] = mads_constrained(func,constr,rho,x0,max_iter,max_fcalls)
    global function_calls;
    function_calls = 0;
    
    %viz kochenderfer 10.30
    f = @(x) func(x) + rho*(constr(x) > 0);
    plot_f(f, [-3,3], [0,0]);
    plot(sqrt(2)*sin(linspace(0,2*pi)),sqrt(2)*cos(linspace(0,2*pi)),'k-');
    title('stochastic or population method - mads');

    %mads prevzat z g-drive
    epsilon = 1e-3;
    [xs,iter] = mads(f,x0,epsilon,max_iter,max_fcalls);
    plot(xs(1,:),xs(2,:),'r*-');

    x0 = xs(:,end);
    fprintf('mads took: \n iterations = %d \n fcalls = %d \nwith best found at: [%f %f]\n', iter, function_calls, x0(1), x0(2))
end
%% BFGS q-n
function [] = BFGS_qn_constrained(func,func_grad,constr,constr_grad,rho,x0,max_iter,max_fcalls)
    global function_calls;
    function_calls = 0;
    total_iter = 0;
    for k=1:length(rho)
        clf;
        %viz kochenderfer 10.32
        f = @(x) func(x) + rho(k)*max(constr(x),0)^2;
        f_grad = @(x) func_grad(x) + rho(k)*2*max(constr(x),0).*constr_grad(x);
        plot_f(f, [-3,3], [0,0]); 
        plot(sqrt(2)*sin(linspace(0,2*pi)),sqrt(2)*cos(linspace(0,2*pi)),'k-');
        title('BFGS Quasi-Newton');
    
        %prevzato z g-drive cv4.m
        g = f_grad(x0);
        xs = [x0];
        iter = 1;
        Q = eye(2);
        min_norm = 1e-2; 
        max_iter = 1e1;
        while (norm(g) > min_norm) && (function_calls < max_fcalls) && (iter < max_iter)
            d = -g;
            if iter == 1
                d = -g;
            else
                gamma = g - g_old;
                delta = xs(:,end) - xs(:,end-1);
                Q = Q - (delta*gamma'*Q + Q*gamma*delta')/(delta'*gamma) + (1+ gamma'*Q*gamma/(delta'*gamma))*delta*delta'/(delta'*gamma);
                d = -Q*g;
            end
            f_red = @(alpha) f(xs(:,end) + d*alpha);
            [a,c] = bracket_minimum(f_red);
            [a,b] = golden_section_search(f_red,a,c,10);
            alpha = (a+b)/2;
            x_new = xs(:,end) + alpha*d;
            xs(:,end+1) = x_new;
            g_old = g;
            g = f_grad(x_new);
            iter = iter + 1;

            plot(xs(1,:),xs(2,:),'r-*');
        end
        x0 = xs(:,end);
        total_iter = total_iter + iter;
        pause(1);
    end
    fprintf('BFGS q-n took: \n iterations = %d \n fcalls = %d \nwith best found at: [%f %f]\n', total_iter, function_calls, x0(1), x0(2))
end
%% BFGS q-n with Augmented Lagrange
function [] = BFGS_qn_lagrange_constrained(func,func_grad,constr,constr_grad,rho,x0,max_iter,max_fcalls)
    global function_calls;
    function_calls = 0;
    total_iter = 0;
    lambda = 0;
    for k=1:length(rho)
        clf;
        %viz chatGPT - explain Augmented Lagrange with equality constraint
        f = @(x) func(x) + lambda*constr(x) + 0.5*rho(k)*(constr(x)^2);
        f_grad = @(x) func_grad(x) + lambda*constr_grad(x) + rho(k)*(constr(x))*constr_grad(x);
        plot_f(f, [-3,3], [0,0]);
        plot(sqrt(2)*sin(linspace(0,2*pi)),sqrt(2)*cos(linspace(0,2*pi)),'k-');
        title('BFGS - Augmented Lagrange');
    
        g = func_grad(x0);
        xs = [x0]; iter = 1;
        Q = eye(2);
        min_norm = 1e-2;
        max_iter = 1e1;
        while (norm(g) > min_norm) && (function_calls < max_fcalls) && (iter < max_iter)
            d = -g;
            if iter == 1
                d = -g;
            else
                gamma = g - g_old;
                delta = xs(:,end) - xs(:,end-1);
                Q = Q - (delta*gamma'*Q + Q*gamma*delta')/(delta'*gamma) + (1+ gamma'*Q*gamma/(delta'*gamma))*delta*delta'/(delta'*gamma);
                d = -Q*g;
            end
            f_red = @(alpha) f(xs(:,end) + d*alpha);
            [a,c] = bracket_minimum(f_red);
            [a,b] = golden_section_search(f_red,a,c,3);
            alpha = (a+b)/2;
            x_new = xs(:,end) + alpha*d;
            xs(:,end+1) = x_new;
            g_old = g;
            g = f_grad(x_new);
            iter = iter + 1;
            
            plot(xs(1,:),xs(2,:),'r-*');
        end
        x0 = xs(:,end);
        lambda = lambda + rho(k)*constr(x0);
        total_iter = total_iter + iter;
        pause(1);
    end
    fprintf('BBFGS q-n with Augmented Lagrange took: \n iterations = %d \n fcalls = %d \nwith best found at: [%f %f]\n', total_iter, function_calls, x0(1), x0(2))
end
%% other
function [f] = flower(x)
    global function_calls;
    function_calls = function_calls + 1;
    a = 1;
    b = 1;
    c = 4;
    r = sqrt(x(1).^2 + x(2).^2);
    theta = atan2(x(2), x(1));
    f = a * r + b * sin(c * theta);
end

function [direction] = flower_grad(x)
    global gradient_calls
    gradient_calls = gradient_calls + 1;
    a = 1;
    b = 1;
    c = 4;

    r = sqrt(x(1).^2 + x(2).^2);
    theta = atan2(x(2), x(1));

    dr_dx1 = x(1) / r;
    dr_dx2 = x(2) / r;

    dtheta_dx1 = -x(2) / (x(1)^2 + x(2)^2);
    dtheta_dx2 = x(1) / (x(1)^2 + x(2)^2);

    direction = [a * dr_dx1 + b * c * cos(c * theta) * dtheta_dx1; a * dr_dx2 + b * c * cos(c * theta) * dtheta_dx2];
end

function plot_f(f, range, opt)
    global function_calls
    xs = linspace(range(1),range(2));
    ys = linspace(range(1),range(2));
    [Xs,Ys] = meshgrid(xs,ys);
    for i=1:100
        for j=1:100
            Zs(i,j) = f([Xs(i,j),Ys(i,j)]);
        end
    end
    contour(Xs,Ys,Zs); hold on; grid on; axis equal;
    plot3(opt(1),opt(2),0,'k*');
    function_calls = function_calls - 10000;
end

function [xs,iter] = cyclic_coordinate_descent(f,x0,par_eps,max_iter,max_fcalls)
    global function_calls;
    xs = [x0]; iter = 1; Delta = inf; n = length(x0); D = eye(n);
    nr_f_calls_line_search = 7;
    acceleration_step = true;
    while abs(Delta) > par_eps
        x_1 = xs(:,end);
        for i=1:n
            d = D(:,i);
            f_red = @(alpha) f(xs(:,end)) + f(xs(:,end)+d*alpha);
            [a,c] = bracket_minimum(f_red);
            [a,b] = golden_section_search(f_red,a,c,nr_f_calls_line_search);
            alpha = (a+b)/2; x_new = xs(:,end) + alpha*d;
            xs(:,end+1) = x_new; 
        end
        if acceleration_step
            d = xs(:,end) - x_1;
            f_red = @(alpha) f(xs(:,end)) + f(xs(:,end)+d*alpha);
            [a,c] = bracket_minimum(f_red); [a,b] = golden_section_search(f_red,a,c,nr_f_calls_line_search);
            alpha = (a+b)/2; x_new = xs(:,end) + alpha*d; xs(:,end+1) = x_new; 
            Delta = norm(xs(:,end)-x_1);
            iter = iter + 1;
        end
        if iter > max_iter || function_calls> max_fcalls
           break; 
        end
    end
end

function [xs,iter] = mads(f,x0,par_eps,max_iter,max_fcalls)
    global function_calls;
    xs = [x0]; 
    alpha = 1; y = f(x0); n = length(x0); x = x0;
    iter = 1;
    while alpha > par_eps
        improved = false;
        D = rand_positive_spanning_set(alpha,n);
        for i=1:3
            d = D(:,i); x_p = x+alpha*d; y_p = f(x_p);
            if y_p < y
                x = x_p; y = y_p; improved = true;
                x_p = x+3*alpha*d; y_p = f(x_p);
                if y_p < y
                    x = x_p; y = y_p;
                end
                break;
            end
        end
        if improved
            alpha = min(4*alpha,1);
        else
            alpha = alpha/4;
        end
        xs(:,end+1) = x;
        iter = iter + 1;
        if iter > max_iter || function_calls > max_fcalls
            break;
        end
    end
end

function [D] = rand_positive_spanning_set(alpha,n)
    delta = round(1/sqrt(alpha));
    L = diag(delta*(2*rand(n,1)-1));
    for i=1:n-1
        for j=i+1:n
            L(i,j) = -delta+1 +(2*delta+2)*rand;
        end
    end
    
    D = L; D = D(randperm(n),:); D = D(:,randperm(n));
    D = [D, -sum(D,2)];
end

% trocha inspirace prevzata z g-drive/Matlab scripty/CV4/cv6.m
% asi jste tam omylem nahral neco navic misto funkce pro flower fcn :)