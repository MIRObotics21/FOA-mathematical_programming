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

