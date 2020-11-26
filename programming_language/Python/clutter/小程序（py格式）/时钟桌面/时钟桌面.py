import pygame
from pygame.locals import *
import datetime
import math
import random
class ball():
    def __init__(self):
        self.R = random.randint(300, 800)  # 距离中心点半径
        self.rot_angle = 0.5
        self.pre_angle = random.randint(1, 360)
        self.location = [0, 0]
        self.r = random.randint(1, 3)  # 圆半径
        self.t = random.randint(1, 20)

    def move(self):
        self.pre_angle += self.rot_angle
        self.location[0] = int(math.cos((self.pre_angle / 360) * 2 * math.pi) * self.R + box[0] / 2)
        self.location[1] = int(math.sin((self.pre_angle / 360) * 2 * math.pi) * self.R + box[1] / 2)

def word(text, size=10, color=(255, 222, 173), bold=0):
    # Fonts = pygame.font.get_fonts()
    fofont_surface = []
    for each in text:
        font = pygame.font.SysFont('SimHei', size)
        font.set_bold(bold)
        fofont_surface.append(font.render(each, True, color))
    return fofont_surface

def transform(char, st):
    newchars = []
    for each in char:
        each = str(each)
        if st == 'w':
            trantab = str.maketrans('1234567', '一二三四五六日')
            newchars.append(('周' + each.translate(trantab).replace('0', '')).ljust(2, '　'))
        else:
            if len(each) == 2:
                each = each[0] + ' ' + each[1]
            trantab = str.maketrans('123456789 ', '一二三四五六七八九十')
            newchars.append((each.translate(trantab).replace('0', '') + st).ljust(4, '　'))
    return newchars


def creat_time():  # 时间生成————————————————
    date = datetime.datetime.now()
    week = date.isoweekday()
    year, month, hour, = date.year, date.month, date.hour
    hour = hour - 12 if hour >= 12 else hour
    next_month = datetime.date(year + 1 if (month == 12) else year, 1 if (month == 12) else month + 1, 1)
    this_month_days = -datetime.date(year, month, 1).__sub__(next_month).days  # 计算当月天数
    datenow = [date.month, date.day, week, hour, date.minute, date.second, date.year, this_month_days]
    return datenow

def creat_surface(Col):  # Surface生成————————————————
    tr_month = word(transform(range(1, 13), '月'),color=Col)
    tr_day = word(transform(range(1, 32), '日'),color=Col)
    tr_hour = word(transform(range(1, 13), '点', ),color=Col)
    tr_minute = word(transform(range(1, 60), '分') + [' '],color=Col)
    tr_second = word(transform(range(1, 60), '秒') + [' '],color=Col)
    tr_week = word(transform(range(1, 8), 'w'),color=Col)
    return [tr_month, tr_day, tr_week, tr_hour, tr_minute, tr_second]

def show():
    datesurface = creat_surface(Col=(255, 222, 173))
    now_datesurface=creat_surface(Col=(255, 100, 1))
    balllist = [ball() for i in range(200)]
    while True:
        datenow = creat_time()
        this_month_days = datenow[-1]
        list_angle = [-360 / 12, -360 / this_month_days, -360 / 7, -360 / 12, -360 / 60, -360 / 60]
        datesurface[1] = datesurface[1][:this_month_days]  # 当月天数调整
        screen.blit(scr2, (0, 0))
        ticks = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                exit()
        m = [80,120,150,180,220,260]
        for i in range(0, len(datesurface)):
            angle = 0
            for j in range(0,len(datesurface[i])):
                temp_angle=angle-(datenow[i]-1)*list_angle[i]
                temp= now_datesurface[i][j] if int(temp_angle)==0 else datesurface[i][j]
                x = int(box[0] / 2 + math.cos((temp_angle  / 180) * math.pi) * m[i])
                y = int(box[1] / 2 - math.sin((temp_angle  / 180) * math.pi) * m[i])
                temp = pygame.transform.rotate(temp, temp_angle)
                rect = temp.get_rect()
                rect.center = x, y
                angle += list_angle[i]
                screen.blit(temp, rect)

        for each in balllist:
            pygame.draw.circle(screen, (100, 200, 233), each.location, each.r)
            each.move()
        yearsur = word([str(datenow[-2])], size=30, color=(255, 255, 255), bold=3)
        yearcect = yearsur[0].get_rect()
        yearcect.center = int(box[0] / 2), int(box[1] / 2)
        screen.blit(yearsur[0], yearcect)
        clock.tick(60)
        pygame.time.wait(100)
        pygame.display.update()

if __name__ == '__main__':
    pygame.init()
    box = (1366, 768)
    # box = (1920, 1080)
    pygame.display.set_caption("时钟特效")
    screen = pygame.display.set_mode(box, FULLSCREEN, 32)
    clock = pygame.time.Clock()
    scr2 = pygame.Surface(box, flags=pygame.SRCALPHA)
    scr2.fill((0, 0, 0, 40))
    sound = pygame.mixer.music.load(r'国潮时代伴奏.mp3') #去掉注释此两行为音乐，
    pygame.mixer.music.play() #去掉注释此两行为音乐，
    show()

