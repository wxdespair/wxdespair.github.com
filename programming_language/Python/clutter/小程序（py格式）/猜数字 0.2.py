print('…………我的…………')
temp = input("猜猜现在我想到的是哪个数字：")
guess = int(temp)
if guess == 8:
    print("对了")
    print("对了就对了吧")
else :
    if guess > 8:
         print("大了")
    else :
        print("小了")
print("不玩了")
