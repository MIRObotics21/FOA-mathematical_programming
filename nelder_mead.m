function [xs,iter,S_s] = nelder_mead(f,x0,S, par_eps,max_iter,max_fcalls)
global function_calls;
xs = [x0]; 
Delta = inf; 
[n,m] = size(S);
alpha = 1; beta = 2; gamma = 0.2; iter = 1; 
S_s = S;
for i=1:m
    y(1,i) = f(S(:,i));
end
while Delta > par_eps
    [y,p] = sort(y); S = S(:,p);
    x_l = S(:,1); y_l = y(1); x_h = S(:,end); y_h = y(end); x_s = S(:,end-1); y_s = y(end-1); x_m = mean(S(:,1:end-1),2); x_r = x_m + alpha*(x_m - x_h); y_r = f(x_r);
    if y_r < y_l
        x_e = x_m + beta*(x_r-x_m); y_e = f(x_e);
        if y_e < y_r 
            S(:,end) = x_e; y(end) = y_e;
        else
            S(:,end) = x_r; y(end) = y_r;
        end
    elseif y_r >= y_s
        if y_r < y_h
            x_h = x_r; y_h = y_r; S(:,end) = x_r; y(end) = y_r;
        end
        x_c = x_m + gamma*(x_h - x_m);
        y_c = f(x_c);
        if y_c > y_h
            for i=2:length(y)
                S(:,i) = (S(:,i)+x_l)/2;
                y(i) = f(S(:,i));
            end
        else
            S(:,end) = x_c; y(end) = y_c;
        end
    else
        S(:,end) = x_r; y(end) = y_r;
    end
    Delta = std(y);
    xs(:,end+1) = x_l;
    iter = iter + 1; 
    S_s(:,:,iter) = S;
    if iter > max_iter || function_calls > max_fcalls
        break;
    end
end
end

