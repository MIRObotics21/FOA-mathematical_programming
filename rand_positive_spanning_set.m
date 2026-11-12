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

