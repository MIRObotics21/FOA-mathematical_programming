clear;clc;close all;
%vetsi - zustava, mensi meni znamenko, pokud neni jde +/-
%pro omezeni to same plus pokud je rovnost nic, pokud >= odecita se x

%krit f >>> otocene hodnoty, kvuli tomu ze maximalizujete
c_z = [-2; 3; 4; -3; 3; 0; 0];
%omezeni
A_z = [-2 -3 0 1 -1 0  0;
       -1 0 -2 0 0 -1 0;
       -3 -1 1 0 0 0 -1];
%prave strany
b = [5; -6; 2;];

[m,n] = size(A_z);

E = eye(m);
for i=1:m
    if b(i) < 0
        E(i,i) = -1;
    end
end

A_p1 = [A_z,E];
b_p1 = b;
c_p1 = [0*c_z;ones(m,1)];

B_ids = [n+1:n+m]; N_ids = [1:n];


[x,B_ids,N_ids,zs,flag] = simplex_method(c_p1,A_p1,b_p1,B_ids,N_ids);

any(B_ids >= n+1);
if (any(B_ids >= n+1))
    disp("neresitelne")
    return
end
%pokud zs = 0 je reseni
zs(end)
disp("zs je nula jedem dal")

N_ids = setdiff(1:n,B_ids);

[x,B_ids,N_ids,zs,flag] = simplex_method(c_z,A_z,b,B_ids,N_ids);
%vysledek
zs(end)

%optional kontrola
%[x_lp,fval_lp] = linprog(c_z,[],[],A_z,b,zeros(8,1))

%zbytek zkopirovany od Vas
function [x,B_ids,N_ids,zs,flag] = simplex_method(c,A,b,B_ids,N_ids)
    [m,n] = size(A);
    B = A(:,B_ids); N = A(:,N_ids);
    vareps = -1e-6;
    x_B = B\b;
    x = zeros(n,1); x(B_ids) = x_B;
    iter = 1; zs(iter) = c'*x; flag = 0;
    while true
        lambda = B'\c(B_ids);
        mu_N = c(N_ids) - N'*lambda;
        if all(mu_N >= vareps)
            flag = 1; 
            display('optimum found');
            break;
        end
        [~,q_ind] = min(mu_N);
        q = N_ids(q_ind);
        d = B\A(:,q);
        if all(d <= 0)
            flag = 2; %unbounded
            display('unbounded');
            iter = iter + 1;
            zs(iter) = -Inf;
            break;
        end
        d_i_positive = find(d > 0);
        ratios = x(B_ids(d_i_positive)) ./ d(d_i_positive);
        [x_q_plus, p_idx] = min(ratios);
        p = B_ids(d_i_positive(p_idx));
        x(B_ids) = x(B_ids) - d*x_q_plus;
        x(N_ids) = 0;
        x(q) = x_q_plus;
        B_ids = [B_ids(B_ids ~= p),q];
        N_ids = [N_ids(N_ids ~= q),p];
        B = A(:,B_ids); N = A(:,N_ids);
        iter = iter + 1;
        zs(iter) = c'*x;
    end
end