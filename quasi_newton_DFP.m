function x_history = quasi_newton_DFP()
    % Parameters
    a = 1; 
    b = 5; 
    tol = 1e-2; % Termination when norm(∇f) < tol
    max_iter = 100; % Maximum iterations
    x = [-1; -1]; % Initial point
    n = length(x);
    Q = eye(n); % Initial Hessian approximation (Identity matrix)
    x_history = [x];
    
    for k = 1:max_iter
        % Compute gradient
        g = grad_rosenbrock(x, a, b);
        
        % Check stopping criterion
        if norm(g) < tol
            break;
        end
        
        % Compute search direction
        p = -Q * g;
        
        % Line search for optimal step size α
        alpha = line_search(@rosenbrock, x, p, a, b);
        
        % Update x
        s = alpha * p;
        x_new = x + s;
        
        % Compute new gradient
        g_new = grad_rosenbrock(x_new, a, b);
        
        % DFP Update
        y = g_new - g;
        rho = 1 / (y' * s);
        if rho > 0
            Q = Q - (Q * (s * s') * Q) / (s' * Q * s) + (y * y') * rho
        end
        
        % Update x
        x = x_new;
        x_history = [x_history,x];
    end
end

%% Rosenbrock Function
function f = rosenbrock(x, a, b)
    f = (a - x(1))^2 + b * (x(2) - x(1)^2)^2;
end

%% Gradient of Rosenbrock Function
function g = grad_rosenbrock(x, a, b)
    g1 = -2 * (a - x(1)) - 4 * b * x(1) * (x(2) - x(1)^2);
    g2 = 2 * b * (x(2) - x(1)^2);
    g = [g1; g2];
end

%% Line Search (Backtracking)
function alpha = line_search(f, x, p, a, b)
    alpha = 1; % Initial step size
    rho = 0.8; % Reduction factor
    c = 1e-4; % Wolfe condition constant
    
    while f(x + alpha * p, a, b) > f(x, a, b) + c * alpha * grad_rosenbrock(x, a, b)' * p
        alpha = rho * alpha; % Reduce step size
    end
end
