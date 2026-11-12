function [xs,iter,mus,Sigmas,samples_all] = cmaes(f,x0,max_iter,max_fcalls)
global function_calls;
xs = [x0]; iter = 1;  n = length(x0);
m = 4+floor(3*log(n)); m_elite = floor(m/2); 
m = 2; m_elite = 1; 
sigma = 1; Sigma = 10*eye(n); 
ws = zeros(1,m); ws(1:m_elite) = log((m+1)/2) - log(1:m_elite); ws = ws/sum(ws);
mu_eff = 1/sum(ws.^2);
c_sigma = (mu_eff+2)/(n+mu_eff+5);
d_sigma = 1+2*max(0,sqrt((mu_eff-1)/(n+1))-1) + c_sigma;
c_Sigma = (4 + mu_eff/n)/(n + 4 + 2*mu_eff/n);
c_1 = 2/((n+1.3)^2 + mu_eff);
c_mu = min(1-c_1, 2*(mu_eff-2+1/mu_eff)/((n+2)^2 + mu_eff));
E = n^0.5*(1-1/(4*n)+1/(21*n^2));
p_sigma = zeros(2,1); p_Sigma = zeros(2,1); mu = x0; 
mus = x0; Sigmas = Sigma; samples_all = []; samples_elite = [];
fmin = f(x0);
for k=1:max_iter
    samples = mvnrnd(mu,Sigma,m)';
    samples_all(:,:,end+1) = samples;
    for i=1:m
        ys(i) = f(samples(:,i));
    end
    [is,is_perm] = sort(ys,'ascend');
    [minval,minpos] = min(ys);
    if minval < fmin
        fmin = minval;
        xs(:,end+1) = samples(:,minpos);
    end
    delta_s = (samples-mu)/sigma; delta_w = sum(delta_s(:,is_perm).*ws,2);
    mu_new = mu + sigma*delta_w; 
    mu = mu_new; mus(:,end+1) = mu;
    C = Sigma^(-0.5);
    p_sigma = (1-c_sigma)*p_sigma+sqrt(c_sigma*(2-c_sigma)*mu_eff)*C*delta_w;
    sigma = sigma*exp(c_sigma/d_sigma * (norm(p_sigma)/E -1));
    h_sigma = (norm(p_sigma)/sqrt(1-(1-c_sigma)^(2*k)) < (1.4+2/(n+1))*E);
    p_Sigma = (1-c_Sigma)*p_Sigma + h_sigma*sqrt(c_Sigma*(2-c_Sigma)*mu_eff)*delta_w;
    for i=1:m
        if ws(i) >= 0
            w_circ(i) = ws(i);
        else
            w_circ(i) = n*ws(i)/(norm(C*delta_s(:,is_perm(i)))^2);
        end    
    end
    Sigma_sub = zeros(2,2);
    for i=1:m
        Sigma_sub = Sigma_sub + w_circ(i)*delta_s(:,is_perm(i))*delta_s(:,is_perm(i))';
    end
    Sigma = (1-c_1-c_mu)*Sigma + c_1*(p_Sigma*p_Sigma' + (1-h_sigma)*c_Sigma*(2-c_Sigma)*Sigma)+...
        c_mu*Sigma_sub;
    Sigma = triu(Sigma) + triu(Sigma,1)';
    Sigma = 0.5*(Sigma + Sigma');
    Sigmas(:,:,end+1) = Sigma;
    if function_calls > max_fcalls
        break;
    end
    iter = iter + 1;
end
end

