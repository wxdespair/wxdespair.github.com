"""
定义一个计时器的类
start方法启动，stop方法停止
假设计时器对象t1，那么print（t1），和直接调用t1都是显示结果
未启动或停止计时时调用stop方法会有提示
计时器对象可以相加
"""


import time

class Mytimer():
	def __init__(self):
		self.unit = ['年', '月', '天', '时', '分', '秒']
		self.prompt = '未开始计时……'
		self.lasted = []
		self.begin = 0
		self.end = 0

	def __str__(self):
		return self.prompt

	__repr__ = __str__

	def __add__(self,other):
		prompt = '总共运行了'
		result = []
		for index in range(6):
			result.append(self.lasted[index] + other.lasted[index])
			if result[index]:
				prompt += (str(result[index]) + self.unit[index])
		return prompt

	# 开始计时
	def start(self):
		self.begin = time.localtime()
		self.prompt = '提示：请先调用stop()停止计时'
		print("计时开始……")

	# 停止计时
	def stop(self):
		if not self.begin:
			print('提示：请先调用start()开始计时')
		else:
			self.end = time.localtime()
			self._calc()
			print("计时结束……")

	# 内部方法，计算运行时间
	def _calc(self):
		self.lasted = []
		self.prompt = '总共运行了'
		for index in range(6):
			self.lasted.append(self.end[index] - self.begin[index])
			if self.lasted[index]:
				self.prompt += (str(self.lasted[index]) + self.unit[index])
		print(self.prompt)
		# 为下一轮计时初始化变量
		self.begin = 0
		self.end = 0
		
