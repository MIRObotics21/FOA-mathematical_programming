function [xs,iter,mus,Sigmas,samples_all,samples_elite] = cross_entropy_method(f,x0,max_iter,max_fcalls)
global function_calls;
xs = [x0]; n = length(x0);
m = 30; m_elite = 10;
mu = x0; Sigma = 20*eye(n);
mus = x0; Sigmas = Sigma;
y0 = f(x0); fmin = y0;
iter = 1; 
samples_all = [];
samples_elite = [];

for k=1:max_iter
    samples = mvnrnd(mu,Sigma,m)';
    samples_all(:,:,end+1) = samples;
    for i=1:m
        y(i) = f(samples(:,i));
    end
    [y,s] = sort(y,"ascend"); P = samples(:,s(1:m_elite));
    samples_elite(:,:,end+1) = P;
    if y(1) < fmin
        fmin = y(1);
        xs(:,end+1) = P(:,1);
    end
    mu_new = (sum(P,2))/(m_elite);
    mu = mu_new;
    Sigma = cov(P');
    Sigmas(:,:,end+1) = Sigma;
    mus(:,end+1) = mu;
    if function_calls > max_fcalls
        break;
    end
    iter = iter + 1;
end

end

