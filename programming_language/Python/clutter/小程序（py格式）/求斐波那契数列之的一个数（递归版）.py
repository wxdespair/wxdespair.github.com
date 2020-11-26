def fbnq(n):
    if n < 1:
        print("输入有误！")
        return -1

    if n == 1 or n == 2:
        return 1
    else:
        return fbnq(n - 1) + fbnq(n - 2)

num = int(input("欲计算的斐波那契数列的第几位数："))
result = fbnq(num)
if result != -1:
    print("斐波那契数列的第%d个数字是：%d" % (num , result))
