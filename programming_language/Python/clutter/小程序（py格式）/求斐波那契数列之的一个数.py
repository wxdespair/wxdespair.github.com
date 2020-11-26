def fbnq(n):
    n1 = 1
    n2 = 1
    n3 = 1

    if n < 1:
        print("输入有误！")
        return -1

    while (n-2) > 0:
        n3 = n1 + n2
        n1 = n2
        n2 = n3
        n -= 1
    return n3

num = int(input("欲计算的斐波那契数列的第几位数："))
result = fbnq(num)
if result != -1:
    print("斐波那契数列的第%d个数字是：%d" % (num , result))
    
