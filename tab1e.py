import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np
from math import sin, cos, pi, atan2

BALL_RADIUS = 13.48 # 台球标准半径

class Ball:
    def __init__(self, x, y, r, tp):
        self.x, self.y, self.r, self.tp = x, y, r, tp




class Table:
    def __init__(self, loc, size, back, balls, tp='snooker'):
        self.loc, self.size = loc, size
        self.back = back
        self.balls = [Ball(y,x,r,int(tp)) for y, x, r, tp in balls]  # extract模块可以提取出一个r值
        BALL_RADIUS = self.balls[0].r  # 重新设置标准半径
        # self.unit = self.size[1]/100
        self.hitpts = []
        self.vpockets = []  # 可选的袋口坐标
        if tp=='snooker':   
            # 直接硬编码袋口坐标
            self.pockets = [
                (24, 26),      # 左上袋
                (12, 492.5),    # 中上袋
                (24, 959),      # 右上袋
                (496, 959),     # 右下袋
                (509, 492.5),   # 中下袋
                (496, 26)       # 左下袋
            ]

   
    def solve_simple(self, goal=1):
        # cue_ball = next(b for b in self.balls if b.tp == 0)
        targets = [b for b in self.balls if b.tp != 0]
        rst = []


        for px, py in self.pockets:
            
            for ball in targets:
                dx, dy = px - ball.x, py - ball.y
                norm = (dx**2 + dy**2)**0.5
                if norm == 0: continue
                dx, dy = dx / norm, dy / norm
                hit_x = ball.x - dx * BALL_RADIUS * 2
                hit_y = ball.y - dy * BALL_RADIUS * 2
                # cue_dx = hit_x - cue_ball.x
                # cue_dy = hit_y - cue_ball.y
                # cue_dist = (cue_dx**2 + cue_dy**2)**0.5
                # angle = atan2(cue_dy, cue_dx)
                rst.append([hit_x, hit_y,  -1, 1, 0])
        for px, py in self.vpockets:
            print(f"虚拟袋口: {px}, {py}")
            for ball in targets:
                dx, dy = px - ball.x, py - ball.y
                norm = (dx**2 + dy**2)**0.5
                if norm == 0: continue
                dx, dy = dx / norm, dy / norm
                hit_x = ball.x - dx * BALL_RADIUS * 2
                hit_y = ball.y - dy * BALL_RADIUS * 2
                # cue_dx = hit_x - cue_ball.x
                # cue_dy = hit_y - cue_ball.y
                # cue_dist = (cue_dx**2 + cue_dy**2)**0.5
                # angle = atan2(cue_dy, cue_dx)
                rst.append([hit_x, hit_y,  -1, 2, 0])

        self.hitpts = np.array(rst)

    def show(self):
        plt.imshow(self.back)
        angs = np.linspace(0, np.pi*2, 36)
        rs, cs = np.cos(angs), np.sin(angs)
        lut = np.array([(255,255,255), (255,0,0),
            (255,255,0), (0,255,0), (128,0,0),
            (0,0,255), (255,128,128), (50,50,50)])/255
        for i in self.balls:
            plt.plot(cs*i.r+i.y, rs*i.r+i.x, c=lut[i.tp])
        h, w = self.size
        plt.plot([0, 0, w, w, 0], [0, h, h, 0, 0], 'blue')
        for line in self.pockets:
            r1, c1, r2, c2 = line.x1, line.y1, line.x2, line.y2
            plt.plot([c1,c2], [r1,r2], 'white')
        if len(self.hitpts)>0:
            plt.plot(self.hitpts[:,1], self.hitpts[:,0], 'r.')
        plt.show()
