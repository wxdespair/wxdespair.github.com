% 格式化输出的交互温度转化与交互加法
C = input('Enter a temperature in Celsius:');
F = (C * 1.8) + 32;
fprintf('Fahrenheit = %f\n', F);
x = input('Please enter x:');
y = input('Please enter y:');
fprintf('%f + %f = %f\n', x, y, x + y);
C = input('Enter a temperature in Celsius:');
F = (C * 1.8) + 32;
fprintf('Fahrenheit = %g\n', F);						% 格式化输出时写g即可去掉多余的0
x = input('Please enter x:');
y = input('Please enter y:');
fprintf('%g + %g = %g\n', x, y, x + y);					% 格式化输出时写g即可去掉多余的0