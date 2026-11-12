function p = cov_mat_plot(mu,Sigma)

coef = 3;
[vec,val] = eig(Sigma);
t = linspace(0,2*pi,100);
circle = [cos(t);sin(t)];
ellipse = coef*vec'*sqrt(val)*vec*circle + mu;
p = plot(ellipse(1,:),ellipse(2,:),'-g','LineWidth',3);

end