function result = test_function_3(n)
    count = 0;
    for i = 1:n
        if mod(n, i) == 0
            count = count + 1;
            % disp(i);
        end
    end
    result = count;                 % ������result����������ֵ���������ں�����ǰָ��
end