% 计算两个数的最大公约数
a = input('a = ');
b = input('b = ');

r = mod(a,b);
while r ~= 0
    a = b;
    b = r;
    r = mod(a, b);
end
disp(b)