clear;clc;
load prob1.mat;
f = @(x) simulator(x);
g = @(x) norm(A*x - b, 2) <= 1;
rho1 = 10; %muze zustat
rho2 = 10;

f_pen = @(x) f(x) + rho1*(g(x)>0) + rho2*(max(g(x),0)^2);
f_pen_sat = @(x) f_pen(min(max(x,LB),UB));

%vhodne vybrat pocatek
x0 = (LB+UB)/2;

max_iter = 100;
max_fcalls = 1e7;

%upravit podle toho ktere metody bude vyzadovat
[xs] = diff_evol(f,x0,max_iter,max_fcalls);
x0 = xs(:,end);
fprintf('nejlepsi x po diff evoluci found at: [%f %f %f %f %f] \n', x0(1),x0(2),x0(3),x0(4),x0(5))
%splneni podminky
g(min(max(xs(:,end),LB),UB));
%funkcni hodnota
fprintf('jeho fkcni hodnota: %d \n\n', f(min(max(xs(:,end),LB),UB)))

par_eps = 1e-6;
max_iter = 100;
max_fcalls = 1e7;

[xs] = mads(f,x0,par_eps,max_iter,max_fcalls);
fprintf('nejlepsi x po MADSU found at: [%f %f %f %f %f] \n', x0(1),x0(2),x0(3),x0(4),x0(5))
%splneni podminky
g(min(max(xs(:,end),LB),UB));
%funkcni hodnota
fprintf('jeho fkcni hodnota: %d \n\n', f(min(max(xs(:,end),LB),UB)))

disp('ten mads tu mam nejak slaby - obcas zlepsi ale vetsinou nic, nenasel sem chybu')


%% DIFF EVOLUCE UKRADENA OD VAS
function [xs,iter,pops] = diff_evol(f,x0,max_iter,max_fcalls)
    function_calls = 0;
    n = length(x0); 
    m = 50; 
    w = 0.8; p = 0.9;
    iter = 1;
    xs = x0; fmin = f(x0);
    pop = [x0+randn(n,m)]; 
    pops = pop;
    for i=1:m
        val(i) = f(pop(:,i));
    end
    for k=1:max_iter
        children = zeros(5,m);
        for i=1:m
            perm = randperm(m,3);
            z = pop(:,perm(1)) + w*(pop(:,perm(2)) - pop(:,perm(3)));
            j = randperm(n,1);
            children(:,i) = pop(:,i);
            for d = 1:n
                if d == j || p < rand(1)
                    children(d,i) = z(d);
                end
            end
        end
        for i=1:m
            val_child = f(children(:,i));
            if val_child < val(i)
                pop(:,i) = children(:,i);
                val(i) = val_child;
                if val_child < fmin
                    fmin = val_child;
                    xs(:,end+1) = children(:,i);
                end
            end
        end
        pops(:,:,end+1) = pop;
        if function_calls > max_fcalls
            break;
        end
        iter = iter + 1;
    end
end

%% MADS UKRADENY OD VAS
function [xs,iter] = mads(f,x0,par_eps,max_iter,max_fcalls)
    function_calls = 0;
    xs = [x0]; 
    alpha = 1; y = f(x0); n = length(x0); x = x0;
    iter = 1;
    while alpha > par_eps
        improved = false;
        D = rand_positive_spanning_set(alpha,n);
        for i=1:6
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


