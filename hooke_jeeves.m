function [xs,iter,x_ns] = hooke_jeeves(f,x0,par_eps,max_iter,max_fcalls)
global function_calls;
xs = [x0]; iter = 1; n = length(x0);
D = eye(n); y = f(xs(:,end)); 
gamma = 0.5; alpha_0 = 1.2;
alpha = alpha_0;
do_restart = false;
x_ns = [];
while alpha > par_eps || do_restart
    improved = false; x_best = xs(:,end); y_best = y;
    if alpha < par_eps && do_restart
       alpha = alpha_0;
    end
    cntr = 0;
    for i=1:n
       for sgn = [-1,1]
          x_n = xs(:,end) + sgn*alpha*D(:,i); y_n = f(x_n);
          cntr = cntr + 1;
          x_ns(:,cntr,iter) = x_n;
          if y_n < y_best
             x_best = x_n; y_best = y_n; improved = true; 
          end
       end
    end
    xs(:,end+1) = x_best; y = y_best; 
    if ~improved
       alpha = alpha*gamma; 
    end
    if iter > max_iter || function_calls> max_fcalls
       break; 
    end
    iter = iter + 1;
end


end

