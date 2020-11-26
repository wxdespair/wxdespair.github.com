下述代码运行后即可得出结果



求逆矩阵：

```python
import numpy as np
d = np.array(((6,3,-1),(-1,3,-7),(10,12,2)))
final = np.linalg.inv(d)
print(final)
# d为矩阵，后方行向量组为表示的矩阵
```



有2*3的矩阵D，其依次包含下列元素：
    (0   1   4)
    (4   16   6)
在 python中通过numpy.sqrt()求矩阵开方。求矩阵D所有元素数值开根号，将结果保存在D_sqrt中.
正误判定变量：D_sqrt

```python
import numpy as np
D= np.array(((0,1,4),(4,16,6)))
D_sqrt=np.sqrt(D)
print(D_sqrt)
```



有一个2*2的矩阵D，其依次包含下列元素：
    (3   4)
    (6   8)
在python 中通过numpy.square计算平方。求矩阵D所有元素数值的平方值，将结果保存在D_square中
正误判定变量：D_square

```python
import numpy as np
D= np.array(((3, 4),(6,8)))
D_square=np.square(D)
print(D_square)
```



已知2*3的矩阵D，其依次包含下列元素：
    (0   -1   4)
    (9   16   -25)
在python中通过numpy.abs()求绝对值。求矩阵D所有元素数值的绝对值，将结果保存在D_abs中
正误判定变量：D_abs

```python
import numpy as np
D= np.array(((0,-1,4), (9, 16, -25)))
D_abs=np.abs(D)
print(D_abs)
```



已有矩阵
    (0   1)
    (2   3)
在python中通过numpy.transpose()求转置矩阵。求上述矩阵的转置,结果保存到final中

```python
import numpy as np
a = np.array([[0, 1],[2, 3]])
final = np.transpose(a)
print(final)
```



已知矩阵A，其依次包含下列元素：
    (1   2   3)
    (4   5   6)
在python可以通过numpy.linalg.svd()函数实现矩阵的SVD处理。计算矩阵A的奇异值分解。
正误判定变量：final

```python
import numpy as np
A=np.mat([[1,2,3],[4,5,6]])
U,Sigma,VT = np.linalg.svd(A)
print(U)
print('----分割线-------')
print(Sigma)
print('----分割线-------')
print(VT)
```



已知在python中可以通过sympy库的limit(f(x),x,x0)函数表示  x→x0  时  f(x)  的取值，即  limx→x0f(x) （其中  ∞  在sympy中用两个小写字母“o”表示）
利用python求  limx→0sin(x)/x 的值，最终结果保存在final中。

```python
from sympy import *
x, y, z = symbols('x y z')
init_printing(use_unicode=True)
final = limit(sin(x)/x,x,0)
print(final)
```



x\*\*2/exp(x)表示x^2/e^x  ，  exp(x\*y)表示e^(x\*y)   ， pi表示π

已知在python中可以通过sympy库的diff(f(x),x)函数对  f(x)  进行求导
利用python求 cos(x) 的导数，结果保存在final中。

```python
from sympy import *
x, y, z = symbols('x y z')
init_printing(use_unicode=True)
final = diff(cos(x),x)
print(final)
```



已知在python中可以通过sympy库的Derivative(f(x,y,z),x).doit()函数对  f(x,y,z)  的变量  x  求偏导，
利用python求  e^(x*y)  对  x  的偏导数，最终结果保存在final中。

```python
from sympy import *
x, y, z = symbols('x y z')
init_printing(use_unicode=True)
final = Derivative(exp(x*y),x).doit()
print(final)
```



已知在python中可以通过sympy库的f(x).series(x,x0,n)函数表示  f(x)  的泰勒展开，（其中 x0 若无指定默认取值为 0 ，n 若无指定默认取值为 6）
利用python求  e^x 的泰勒展开前6项，最终结果保存在final中。

```python
from sympy import *
x, y, z = symbols('x y z')
init_printing(use_unicode=True)
final = exp(x).series(x,0,6)
print(final)
```

