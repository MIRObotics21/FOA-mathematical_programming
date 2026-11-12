function [xs,iter,pops] = pso(f,x0,max_iter,max_fcalls)
global function_calls;
w = 0.9; c_1 = 1.2; c_2 = 1.2; m = 20;
pop = [x0+randn(2,m)]; xs = x0; pops = pop;
v = 0*pop; x_best_i = pop; iter = 1;
for i=1:m
    val(i) = f(pop(:,i));
end
y_best_i = val; [minval, minpos] = min(val); y_best = minval; x_best = pop(:,minpos);
for i=1:m
    v(:,i) = w*v(:,i) + c_1*rand(1)*(x_best_i(:,i) - pop(:,i)) + c_2*rand(1)*(x_best-pop(:,i));
end
for k=1:max_iter
    for i=1:m
        pop(:,i) = pop(:,i) + v(:,i);
        v(:,i) = w*v(:,i) + c_1*rand(1)*(x_best_i(:,i) - pop(:,i)) + c_2*rand(1)*(x_best-pop(:,i));
        val(i) = f(pop(:,i));
        if val(i) < y_best_i(i)
            y_best_i(i) = val(i); x_best_i(:,i) = pop(:,i);
        end
    end
    [minval, minpos] = min(val);
    if minval < y_best
        y_best = minval; x_best = pop(:,minpos);
        xs(:,end+1) = x_best;
    end
    pops(:,:,end+1) = pop;
    if function_calls > max_fcalls
        break;
    end
    iter = iter + 1;
end
end

