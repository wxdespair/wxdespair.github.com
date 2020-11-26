import turtle
turtle.speed(0)
t = turtle.Pen()
t.color("gold")
t.pensize(7)
for x in range(12):
	t.shape("turtle")
	t.circle(100,360)
	t.left(30)

turtle.done()