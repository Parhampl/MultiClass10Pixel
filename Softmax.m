function y = Softmax(x)
    exp_x = exp(x - max(x)); % To avoid numerical overflow
    y = exp_x ./ sum(exp_x);
end
