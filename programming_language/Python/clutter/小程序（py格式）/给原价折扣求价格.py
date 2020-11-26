def discounts(price , rate):
	final_price = price * rate
	old_price = 50
	print('old_price的值1为' , old_price)  #试图修改全局变量
	return final_price
old_price = float(input('请输入原价：'))
rate = float(input('请输入折扣率：'))
new_price = discounts(old_price , rate)
print('old_price的值2为' , old_price)
print('打折后的价格是：' ,new_price)
print('全局变量old_price的值为：' ,old_price )
