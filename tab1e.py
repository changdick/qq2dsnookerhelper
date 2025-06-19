import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np
from math import sin, cos, pi, atan2

BALL_RADIUS = 13.48  # 台球标准半径

class Ball:
    def __init__(self, x, y, r, tp):
        self.x, self.y, self.r, self.tp = x, y, r, tp

class Line:
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2

    def midpoint(self):
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2
    
class Pocket:
    def __init__(self, x,y):
        self.x, self.y = x, y


class Table:
    def __init__(self, loc, size, back, balls, tp='snooker'):
        self.loc, self.size = loc, size
        self.back = back
        self.balls = [Ball(y,x,r,int(tp)) for y, x, r, tp in balls]  # extract模块可以提取出一个r值
        self.unit = self.size[1]/100
        self.hitpts = []
        if tp=='snooker': self.make_pocket(1, 3)
        self.pockets = [
            Pocket(24,26),
            # Pocket(*self.pocket[1].midpoint()),
            Pocket(12, 492.5),
            Pocket(24,959),
            Pocket(496,959),
            Pocket(509,492.5),
            Pocket(496,26)
        ]

    def make_pocket(self, k1, k2):
        unit, h, w = self.unit, *self.size
        mar = unit * k1
        self.pocket = []
        for line in [
            (mar, mar+mar*k2, mar+mar*k2, mar),
            (mar, w/2+mar*k2, mar, w/2-mar*k2),
            (mar+mar*k2, w-mar, mar, w-mar-mar*k2),
            (h-mar, w-mar-mar*k2, h-mar-mar*k2, w-mar),
            (h-mar, w/2-mar*k2, h-mar, w/2+mar*k2),
            (h-mar-mar*k2, mar, h-mar, mar+mar*k2),
        ]:
            self.pocket.append(Line(*line))

    def solve_simple(self, goal=1):
        # cue_ball = next(b for b in self.balls if b.tp == 0)
        targets = [b for b in self.balls]
        rst = []

        for pocket in self.pockets:
            px, py = pocket.x, pocket.y
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
                rst.append([hit_x, hit_y,  -1, 0, 0])

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
