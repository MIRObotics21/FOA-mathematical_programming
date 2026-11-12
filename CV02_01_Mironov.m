clc; clear; close all;

%zadani
a = 1;
b = 5;
rosenbrock = @(x1, x2) (a - x1).^2 + b * (x2 - x1.^2).^2;
x0 = [-1; -1];
max_iter = 100;
max_grad = 1e-2;
max_eval = 50;

%k zadani pridam funkci pro gradient a vstupy pro plot rosenbrocka, jelikoz ji budu vsude pouzivat
grad_rosenbrock = @(x1, x2) [-2*(a - x1) - 4*b*x1*(x2 - x1.^2); 2*b*(x2 - x1.^2)];
[x1_vals, x2_vals] = meshgrid(linspace(-2, 2, 100), linspace(-2, 2, 100));
f_vals = rosenbrock(x1_vals, x2_vals);
f = @(x) (a - x(1))^2 + b*(x(2) - x(1)^2)^2;
grad_f = @(x) [-2*(a - x(1)) - 4*b*x(1)*(x(2) - x(1)^2); 2*b*(x(2) - x(1)^2)];



%Gradient method with decaying stepsize

%plot rosenbrockovi fce
figure(1);
hold on;
contour(x1_vals, x2_vals, f_vals, 50, 'LineWidth', 1, 'LineColor', 'k');
xlabel('x1'); ylabel('x2');
title('Gradient method with decaying stepsize');

%fce pro decaying stepsize
x = x0;
x_history = x;
for k = 1:max_iter
    grad = grad_rosenbrock(x(1),x(2));
    grad_normalised = grad/norm(grad);
    alpha = 0.9^k;
    x_new = x - alpha * grad_normalised;
    x_history = [x_history, x_new];
    x = x_new;
    if norm(grad_normalised) < max_grad
        break;
    end
end

%plot iteraci
x1 = x_history(1,:);
x2 = x_history(2,:);
plot(x1, x2, 'r-', 'LineWidth', 1)
hold off




%Gradient method with optimal stepsize α obtained from gsearch

%plot rosenbrockovi fce
figure(2);
hold on;
contour(x1_vals, x2_vals, f_vals, 50, 'LineWidth', 1, 'LineColor', 'k');
xlabel('x1'); ylabel('x2');
title('Gradient method with optimal stepsize α obtained from gsearch');

function [optimal_stepsize] = gsearch_bracket(f, x, grad, lower, upper, max_evals)
    phi = (1 + sqrt(5)) / 2;
    iter = 0;
    
    while iter < max_evals
        c1 = upper - (upper - lower) / phi;
        c2 = lower + (upper - lower) / phi;
        
        % Evaluate the function at the new points
        f1 = f(x - c1 * grad);
        f2 = f(x - c2 * grad);
        if f1 < f2
            upper = c2;
        else
            lower = c1;
        end
        iter = iter + 1;
    end
    optimal_stepsize = (upper + lower) / 2;
end

%fce pro optimal stepsize
x = x0;
x_history = x;
for k = 1:max_iter
    grad = grad_rosenbrock(x(1),x(2));
    grad_normalised = grad/norm(grad);
    alpha = gsearch_bracket(f, x, grad_normalised, -2, 2, max_eval);
    x_new = x - alpha * grad_normalised;
    x_history = [x_history, x_new];
    x = x_new;
    if norm(grad_normalised) < max_grad
        break;
    end
end

%plot iteraci
x1 = x_history(1,:);
x2 = x_history(2,:);
plot(x1, x2, 'r-', 'LineWidth', 1)
hold off




%Conjug grad

%plot rosenbrockovi fce
figure(3);
hold on;
contour(x1_vals, x2_vals, f_vals, 50, 'LineWidth', 1, 'LineColor', 'k');
xlabel('x1'); ylabel('x2');
title('Conjugate gradient method with optimal stepsize α obtained from gsearch');

%fce pro conjug grad
x = x0;
x_history = x;
grad = grad_f(x);
d = -grad;
for k = 1:max_iter
    grad = grad_f(x);
    alpha = gsearch_bracket(f, x, -d, -2, 2, max_eval);
    x_new = x + alpha * d;
    grad_new = grad_f(x_new);
    beta = max(0, (grad_new' * (grad_new - grad)) / (grad' * grad));
    d = -grad_new + beta * d;
    x_history = [x_history, x_new];
    x = x_new;
    if norm(grad) < max_grad
        break;
    end
end

%plot iteraci
x1 = x_history(1,:);
x2 = x_history(2,:);
plot(x1, x2, 'r-', 'LineWidth', 1)
hold off