def hnt(n , x , y , z):
    if n == 1:
        print(x , '-->' , z)
    else:
        hnt(n-1 , x , z , y)#将前n-1个从x移动到y上
        print(x , '-->' , z)#将最底下的最后一个从x移动到z上
        hnt(n-1 , y , x , z)#将y上的n-1个移动到z上

n = int(input('请输入汉诺塔的层数：'))
hnt(n , 'X' , 'Y' , 'Z')
input()
