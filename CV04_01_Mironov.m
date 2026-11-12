%----------REQUIREMENTS----------%
%Statistics and Machine Learning Toolbox

clear;clc;clf;
global f_calls;
f = @(x) ackley(x); range = [-8,8]; opt = [0,0]; conts = 0:100;

%globalni parametry zadani
f_calls = 0;
max_iter = 100;
max_fcalls = 1000;
x0 = [-6;-4.5];

%----------MAIN----------%

%mesh_adaptive_direct_search
%lokal zadani
epsilon = 1e-6;

% %body
% x = mesh_adaptive_direct_search(f, x0, epsilon, max_iter, max_fcalls);
% hold on;
% title('Mesh adaptive direct search')
% plot_f(f, range, opt, conts);
% plot(x(1,:),x(2,:),'r-', 'LineWidth', 1)
% hold off

%Cross-entropy method
%lokal zadani
m = 40;
m_elite = 10;
%DISCLAIMER ---->>>> dovolil jsem si sem prenest podminku pro max step
%epsilon z predchozi ulohy, pro deaktivaci odkomentujte dalsi radek
% epsilon = 0;

% %body
% x = cross_entropy_method(f, x0, m, m_elite, epsilon,max_iter, max_fcalls);
% hold on;
% title('Cross-entropy method')
% plot_f(f, range, opt, conts);
% %trosku slozitejsi plot
% set = 1;
% for i = 1:(length(x)/m_elite)
%     to_plot = [];
%     for j = set:(m_elite+set)-1
%         to_plot = [to_plot; [x(j,1) x(j,2)]];
%     end
%     current = plot(to_plot(:,1),to_plot(:,2),'b*');
%     pause(1)
%     if length(x)/m_elite ~= i
%         delete(current)
%     end
%     set = set + m_elite;
% end
% hold off

%Differential evolution
%lokal zadani
pop_size = 10;
p = 0.9;
w = 0.8;

%body
x = differential_evolution(f, x0, pop_size, p, w, max_iter, max_fcalls);
hold on;
title('Differential evolution')
plot_f(f, range, opt, conts);
set = 1;
for i = 1:(length(x)/m_elite)
    to_plot = [];
    for j = set:(m_elite+set)-1
        to_plot = [to_plot; [x(j,1) x(j,2)]];
    end
    current = plot(to_plot(:,1),to_plot(:,2),'b*');
    pause(0.5)
    if length(x)/m_elite ~= i
        delete(current)
    end
    set = set + m_elite;
end
hold off

%Particle swarm optimization
%lokal zadani
pop_size = 10;
w = 0.9;
c1 = 1.2;
c2 = 1.2;

%body
x = particle_swarm_optimization(f, x0, pop_size, w, c1, c2, max_iter, max_fcalls);
hold on;
title('Particle swarm optimization')
plot_f(f, range, opt, conts);
set = 1;
for i = 1:(length(x)/m_elite)
    to_plot = [];
    for j = set:(m_elite+set)-1
        to_plot = [to_plot; [x(j,1) x(j,2)]];
    end
    current = plot(to_plot(:,1),to_plot(:,2),'b*');
    pause(0.5)
    if length(x)/m_elite ~= i
        delete(current)
    end
    set = set + m_elite;
end

%%
function plot_f(f, range, opt, conts)
    global f_calls
    xs = linspace(range(1),range(2));
    ys = linspace(range(1),range(2));
    [Xs,Ys] = meshgrid(xs,ys);
    for i=1:100
        for j=1:100
            Zs(i,j) = f([Xs(i,j),Ys(i,j)]);
        end
    end
    contour(Xs,Ys,Zs,conts); hold on; grid on; axis equal;
    plot3(opt(1),opt(2),0,'r*');
    f_calls = f_calls - 10000;
end

function [val] = ackley(x)
    global f_calls;
    f_calls = f_calls + 1;
    a = 20; b = 0.2; c = 2*pi;
    d = length(x);
    val = -a*exp(-b*sqrt(sum(x.^2))/d) - exp(sum(cos(c*x))/d) + a + exp(1);
end

% viz Kochenderfer
function [D] = rand_positive_spanning_set(alpha, n)
    delta = round(1/sqrt(alpha));
    L = diag(delta * (randi([0, 1], n, 1) * 2 - 1));
    for i = 1:n-1
        for j = i+1:n
            L(i,j) = (2*delta + 2) * rand(1) - delta + 1;
        end
    end
    D = L(:,randperm(n));
    D = D(randperm(n),:);
    D = [D,-sum(D,2)];
end

% viz Kochenderfer
% vraci historii x, pres ktere jsme prosli
% nejlepsi nalezena hodnota by byla mean(poslednich 10ti bodu)
function [x_history] = mesh_adaptive_direct_search(f, x, epsilon, max_iter, max_fcalls)
    global f_calls; 
    f_calls = 0;
    iter = 0;
    alpha = 1;
    y = f(x);
    n = length(x);
    x_history = [x];
    while alpha > epsilon && iter < max_iter && f_calls < max_fcalls
        improved = false;
        D = rand_positive_spanning_set(alpha, n);
        for i = 1:n+1
            d = D(:,i);
            xp = x + alpha * d;
            yp = f(xp);
            if yp < y
                x = xp;
                y = yp;
                improved = true;
                xp = x + 3*alpha * d;
                yp = f(xp);
                x_history = [x_history, x];
                if yp < y
                    x = xp;
                    y = yp;
                    x_history = [x_history, x];
                end
                break
            end
        end
        if improved == true
            alpha = min(4*alpha, 1);
        else
            alpha = alpha/4;
        end
        iter = iter + 1;
    end
    top = x;
    fprintf('mesh adaptive search took: \n iterations = %d \n fcalls = %d \nwith best at: [%f %f]\n', iter, f_calls, top(1), top(2))
end

% viz Kochenderfer
% vraci nejlepsich 10 z kazde iterace, pres ktere jsme prosli, tj. double(iter*10)x2
function [x_history] = cross_entropy_method(f, x, m, m_elite, epsilon, max_iter, max_fcalls)
    global f_calls; 
    f_calls = 0;
    iter = 0;
    x_history = [];
    step = 10;
    % zde jsem si navic pridal podminku nejmensiho kroku 1e-6
    % nebyla soucasti zadani, ale libi se mi
    while iter < max_iter && f_calls < max_fcalls && step > epsilon
        % generovani podle normalu
        samples = mvnrnd(x, step * eye(2), m);
        % ohodnoceni a srovnani podle f(x)
        f_samples = [];
        for i = 1:length(samples)
            f_samples = [f_samples, [f([samples(i,1),samples(i,2)])]];
        end
        [useless_sorted_f_samples, idx] = sort(f_samples);
        samples = samples(idx,:);
        % vycuc top 10
        elite_samples = [];
        for i = 1:m_elite
            elite_samples = [elite_samples; [samples(i,1),samples(i,2)]];
        end
        ny = mean(elite_samples);
        xminusny = elite_samples-[ones(1,m_elite)*ny(1);ones(1,m_elite)*ny(2)]';
        sumik = 0;
        for i = 1:length(m_elite)
            sumik = sumik + xminusny(i,:)*xminusny(i,:)';
        end
        % update vstupnich parametru generovani setu podle Kochenderfera 
        x = x - ny';
        step = sumik/m_elite;
        iter = iter + 1;
        x_history = [x_history; elite_samples];
    end
    top = mean(elite_samples);
    fprintf('cross-entropy method took: \n iterations = %d \n fcalls = %d \nwith best at: [%f %f]\n', iter, f_calls, top(1), top(2))
end

% viz Kochenderfer
% vraci populaci kazde iterace, pres ktere jsme prosli, tj. double(iter*10)x2
function [x_history] = differential_evolution(f, x, pop_size, p, w, max_iter, max_fcalls)
    global f_calls;
    f_calls = 0;
    iter = 0;
    population = mvnrnd(x, 10 * eye(2), pop_size);
    x_history = [population];
    n = length(population(1,:));
    m = length(population(:,1));
    while iter < max_iter && f_calls < max_fcalls
        new_population = [];
        for i = 1:m
            x = population(i,:);
            rand_index = randperm(10, 3);
            a = population(rand_index(1),:);
            b = population(rand_index(2),:);
            c = population(rand_index(3),:);
            z = a + w*(b-c);
            j = randi(1:n);
            for k = 1:n
                if k == j || rand() < p
                    x_new(k) = z(k);
                else
                    x_new(k) = x(k);
                end
            end
            if f(x_new) < f(x)
                x = x_new;
            end
            new_population = [new_population; x];
        end
        population = new_population;
        x_history = [x_history; new_population];
        iter = iter + 1;
    end
    top = mean(new_population);
    fprintf('differential evolution method took: \n iterations = %d \n fcalls = %d \nwith best at: [%f %f]\n', iter, f_calls, top(1), top(2))
end

% viz Kochenderfer
% vraci populaci kazde iterace, pres ktere jsme prosli, tj. double(iter*10)x2
function [x_history] = particle_swarm_optimization(f, x, pop_size, w, c1, c2, max_iter, max_fcalls)
    global f_calls;
    f_calls = 0;
    iter = 0;
    population = mvnrnd(x, 10 * eye(2), pop_size);
    x_history = [population];
    n = length(population(1,:));
    %byl jsem liny pouzivat classy, takze jednotlive vlastnosti x, v a
    %x_best (viz Kochenderfer) jsou zde v listech s odpovidajicimi indexi
    %x_best je x nejlepsiho bodu, ktery dany particle videl
    %y_best funkcni hodnota x_best, setri funkcni volani
    y_best = [];
    x_best = population;
    %celkove nejlepsi nalezene x a jeho funkcni hodnota
    global_y_best = inf;
    global_x_best = [];
    %stejne tak rychlosti
    v = zeros(pop_size,2);
    for i = 1:pop_size
        y = f(population(i,:));
        y_best = [y_best; y];
        if y < global_y_best
            global_y_best = y;
            global_x_best = population(i,:);
        end
    end
    while iter < max_iter && f_calls < max_fcalls
        for i = 1:pop_size
            r1 = rand(1);
            r2 = rand(1);
            population(i,:) = population(i,:) + v(i,:);
            v(i,:) = w*v(i,:) + c1*r1*(x_best(i,:)-population(i,:)) + c2*r2*(global_x_best-population(i,:));
            y = f(population(i,:));
            if y < global_y_best
                global_y_best = y;
                global_x_best = population(i,:);
            end
            if y < y_best(i)
                x_best(i,:) = population(i,:);
            end
        end
        x_history = [x_history; population];
        iter = iter + 1;
    end
    top = mean(population);
    fprintf('Particle swarm optimization took: \n iterations = %d \n fcalls = %d \nwith best at: [%f %f]\n', iter, f_calls, top(1), top(2))
end