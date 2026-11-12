function [val] = wheeler(x)
global function_calls;
function_calls = function_calls + 1;
a = 1.5;
val = -exp(-(x(1)*x(2) - a)^2 - (x(2)-a)^2);
end