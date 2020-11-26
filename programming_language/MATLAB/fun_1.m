function f = fun_1(n)
    if n == 0
        f = 1;
    else
        f = n * fun_1(n-1);
    end
end

