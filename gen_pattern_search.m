function [xs,iter,x_ns] = gen_pattern_search(f,x0,D,par_eps,max_iter,max_fcalls)
global function_calls;
xs = [x0]; iter = 0; n = length(x0);
[~,m] = size(D);
xs = [x0];
alpha = 1; gamma = 0.9; y = f(xs(:,end)); x=x0;
x_ns = [];
while alpha > par_eps && iter < max_iter && function_calls < max_fcalls
    iter = iter + 1; 
    improved = false;
    cntr = 0;
    for i=1:m
        d = D(:,i);
        x_n = x+alpha*d; y_n = f(x_n);
        cntr = cntr + 1;
        x_ns(:,cntr,iter) = x_n;
        if y_n < y
            x = x_n; y = y_n; improved = true;
            D(:,i) = []; D = [d,D];
            break;
        end
    end
    if ~improved
        alpha = alpha * gamma;
    end
    xs(:,end+1) = x; 
end   


end

