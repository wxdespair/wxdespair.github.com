function result = test_function_3(n)
    count = 0;
    for i = 1:n
        if mod(n, i) == 0
            count = count + 1;
            % disp(i);
        end
    end
    result = count;                 % 在这里result代表函数返回值，该名称在函数名前指定
end