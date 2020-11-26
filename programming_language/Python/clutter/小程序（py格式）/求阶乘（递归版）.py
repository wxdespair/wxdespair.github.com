def jc(n):
    if n==1:
        return 1
    else:
        return n *jc(n-1)

num = int(input("请输入一个待求阶乘的整数："))
result = jc(num)
print("%d的阶乘为%d" % (num , result))
