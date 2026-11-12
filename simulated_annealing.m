function [xs,iter,x_ps,x_ps_selected,deltas,probs] = simulated_annealing(f,x0,max_iter,max_fcalls)
global function_calls;
xs = [x0]; iter = 1; n = length(x0); x_ps = x0; 
deltas = []; probs = []; x_ps_selected = 1;
t0 = 100; t = t0; y = f(x0);
x_best = x0; y_best = y; sigma0 = 1; sigma = sigma0;
for k=1:max_iter
    x = xs(:,end);
    x_p = x + sigma*randn(n,1);     y_p = f(x_p);
    x_ps(:,end+1) = x_p;
    delta = y_p - y;
    deltas(end+1) = delta;
    probs(end+1) = exp(-delta/t);
    
    if delta <= 0 || rand() < exp(-delta/t)
        x = x_p; y = y_p;
        x_ps_selected(end+1) = 1;
    else
        x_ps_selected(end+1) = 0;
    end
    if y_p < y_best
        x_best = x_p; y_best = y_p;
    end
    xs(:,end+1) = x_best;
    %t = t0/k; %sigma = sigma0/k;
    %t = t0*(0.95^k); %sigma = sigma0/(k);
    if function_calls > max_fcalls
        break;
    end
    iter = iter + 1;
end


end