def jc(n):
    result = n
    for i in range(1 , n):
        result *= i
		
    return result

num = int(input("请输入一个待求阶乘的正整数："))
result = jc(num)
print("%d 的阶乘是：%d" % (num , result))
