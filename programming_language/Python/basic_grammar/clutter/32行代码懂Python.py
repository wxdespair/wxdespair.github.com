import random		# 导入包/库
"""
函数定义，多行注释
"""
def main():
	print('随机生成学生的分数')
	num_of_student = 20 	# 变量赋值，无需声明
	students = []			# 定义 ‘列表’ 类型
	for i in range(0, num_of_student):
		students.append(random.randint(50, 100))
	statistics(students)

def statistics(stu_list):
	sum = 0
	for stu in stu_list:
		sum = sum + stu
	avg = sum / len(stu_list)
	if avg >= 80:
		print('优秀班级，平均分：%f' % avg)
	else:
		print('忧伤的班级，平均分：%f' % avg)
	print('班级分数：', stu_list)
	max_score = 0
	i = 1
	while i < len(stu_list):
		if stu_list[i] > max_score:
			max_score = stu_list[i]
		i += 1
	print('成绩最好的分数：%d' % max_score)

if __name__ == '__main__':
	main()
	
input()
