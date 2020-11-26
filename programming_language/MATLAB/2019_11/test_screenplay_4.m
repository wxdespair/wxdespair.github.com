% 分支语句 判断正负与三角形的边长的判断
num = input('Please enter a number:');
if num > 0
    fprintf('Positive\n');
elseif num < 0
    fprintf('Negative\n');
else
    fprintf('num = 0\n');
end

a = input('Enter the value of a:'); 
b = input('Enter the value of b:'); 
c = input('Enter the value of c:'); 
if (a + b > c) && (b + c > a) && (a + c > b)
    fprintf('Yes!\n');
else
    fprintf('No!\n');
end
