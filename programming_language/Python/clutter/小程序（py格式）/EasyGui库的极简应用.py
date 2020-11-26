import easygui as g
import sys

while 1:
	g.msgbox('第一个界面小游戏^_^')

	msg = "请问，……………………………………"
	title = "小游戏互动"
	choices = ["1", "2", "3", "4"]

	choice = g.choicebox(msg, title, choices)

	g.msgbox("你的选择是：" + str(choice), "结果")

	msg = "重新开始？"
	title = "请选择"

	if g.ccbox(msg, title):
		pass
	else:
		sys.exit(0)

		
# EasyGui 学习文档【超详细中文版】[增强版] - hunterrea.._
# https://blog.csdn.net/hunterreal/article/details/45478679