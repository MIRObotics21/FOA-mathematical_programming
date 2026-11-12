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
