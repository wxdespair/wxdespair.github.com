% for循环语句
for i = -1 : 5              % 正常情况下前者必须小于后者
    disp(i);
end

for i = 5:-1:-5             % 在中间添加每次循环加的数值（默认为1，即每次加1），才能实现倒循环
    disp(i);
end

% for循环语句实现遍历矩阵
V = [2 3 4 5 6 1 2 3 4 5];
for i = V
    disp(i);
end