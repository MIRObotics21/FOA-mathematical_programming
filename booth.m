function [val] = booth(x)
global function_calls; function_calls = function_calls + 1;
val = (x(1)+2*x(2)-7)^2 + (2*x(1)+x(2)-5)^2;
end

