function [val] = simulator(x)
m = 5; d = length(x); x = x';
val = -sum(sin(x).*(sin([(1:d)/2].*(x.^2)/pi)).^(2*m));
end
