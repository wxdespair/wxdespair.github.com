function result = test_function_2(n)
    s = 0;
    for i = 1:n
        s = s + i;
    end
    result = s;                 % 在这里result代表函数返回值，该名称在函数名前指定
end