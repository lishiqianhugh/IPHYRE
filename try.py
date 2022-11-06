import sys
import time
import pygame
from pygame.locals import *
import pymunk
import pymunk.pygame_util
import matplotlib as mpl
import matplotlib.pyplot as plt
import pymunk.matplotlib_util
import numpy as np
from game_paras import game_paras
from solutions import sol
import pdb
import os
import itertools
from copy import deepcopy


class IPHYRE():
    def __init__(self):
        self.mode = sys.argv[1]
        self.HEIGHT, self.WIDTH = 600, 600
        self.FPS = 60
        self.timestep = 1 / self.FPS
        self.max_time = 5

       
        self.property = []

     

        self.b_mass, self.b_elasticity, self.b_friction = 1.0, 0.1, 0.5
        self.l_friction, self.l_elasticity = 0.5, 0.1
        self.G = (0., 100.0)

        self.space = pymunk.Space()
        self.space.gravity = self.G

        self.solutions = sol

        if self.mode != 'collect':
            pygame.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            self.clock = pygame.time.Clock()
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

    def reset(self):
        for body in self.space.bodies:
            shape = list(body.shapes)[0]
            self.space.remove(shape, shape.body)
        self.screen.fill((255, 255, 255))
        self.add_all()
        self.shape = [1] * len(self.blocks) + [0] * len(self.balls)
    
       

    def add_ball(self, b_pos, radius, mass, elasticity, friction):
        moment = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, moment)
        body.position = b_pos[0], b_pos[1]
        shape = pymunk.Circle(body, radius)
        shape.elasticity = elasticity
        shape.friction = friction
        shape.color = (255, 0, 0, 255)
        self.space.add(body, shape)
        return shape

    def add_static_line(self, l_pos, eli, friction, elasticity):
        static_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        x1, y1, x2, y2 = l_pos[0][0], l_pos[0][1], l_pos[1][0], l_pos[1][1]
        x, y = (x1 + x2) / 2, (y1 + y2) / 2
        static_body.position = x, y
        static_shape = pymunk.Segment(static_body, (x1 - x, y1 - y), (x2 - x, y2 - y), 10)
        static_shape.friction = friction
        static_shape.elasticity = elasticity
        if eli:
            static_shape.color = (164, 164, 164, 255)
        else:
            static_shape.color = (0, 0, 0, 255)
        self.space.add(static_body, static_shape)
        return static_shape

    def add_dynamic_line(self, l_pos, friction, elasticity):
        x1, y1, x2, y2 = l_pos[0][0], l_pos[0][1], l_pos[1][0], l_pos[1][1]
        x, y = (x1 + x2) / 2, (y1 + y2) / 2
        mass = 1.0
        moment = pymunk.moment_for_segment(mass, (0, 0), (0, 0), 10.)
        body = pymunk.Body(mass, moment)
        body.position = x, y
        shape = pymunk.Segment(body, (x1 - x, y1 - y), (x2 - x, y2 - y), 10.)
        shape.friction = friction
        shape.elasticity = elasticity
        self.space.add(body, shape)

    def add_joint(self, b1, b2, a1=(0, 0), a2=(0, 0)):
        joint = pymunk.constraints.PinJoint(b1, b2, a1, a2)
        self.space.add(joint)

    
    def eliminate(self, p):
        for i, body in enumerate(self.space.bodies[:-self.num_ball]):
            shape = list(body.shapes)[0]
            x, y = body.position
            min_0 = x - 10
            max_0 = x + 10
            min_1 = y - 10
            max_1 = y + 10
            if self.eli[i] == 1 and min_0 < p[0] < max_0 and min_1 < p[1] < max_1:
                self.space.remove(shape, shape.body)
                self.eli.pop(i)
                self.dynamic.pop(i)
                self.shape.pop(i)
                return i
        return -1

    def examine_success(self):
        success = 0
        for ball in self.space.bodies[-self.num_ball:]:
            if ball.position[1] > self.HEIGHT:
                success += 1
        if success == self.num_ball:
            return True
        else:
            return False

    def add_text(self, text="Success!", loc=(230, 30), color="green"):
        font = pygame.font.Font(None, 50)
        text = font.render(text, True, pygame.Color(color))
        self.screen.blit(text, loc)
        self.add_restart()

    def add_restart(self, text="Press space to Restart", loc=(200, 200), color="blue"):
        font = pygame.font.Font(None, 30)
        text = font.render(text, True, pygame.Color(color))
        self.screen.blit(text, loc)

    def play(self):
        # p = (200, 190)
        # v = (80, 0)
        # b0 = pymunk.Body(body_type=pymunk.Body.STATIC)
        # b0.position = p
        # radius = 20.
        # b_pos = (p+v)
        # moment = pymunk.moment_for_circle(self.b_mass, 0, radius)
        # body = pymunk.Body(self.b_mass, moment)
        # body.position = b_pos[0], b_pos[1]
        # c = pymunk.Circle(body, radius)
        # self.space.add(b0, c.body)
        # self.add_joint(b0, c.body)
        rotation_center_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        rotation_center_body.position = (300, 300)
        body = pymunk.Body(10, 10000)
        body.position = (300, 400)
        rotation_center_joint = pymunk.PinJoint(body, rotation_center_body, (0, 0), (0, 0))
        self.space.add(body, rotation_center_joint)


        time_count = 0
        while time_count < self.max_time + self.timestep:
            self.screen.fill((255, 255, 255))
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    for b in self.space.bodies:
                        print(len(self.space.bodies))
                        cons = list(b.constraints)[0]
                        self.space.remove(cons)
                    for b in self.space.bodies:
                        print(len(self.space.bodies))
                        print(b)
                        cons = list(b.constraints)[0]
                        print(cons)
                  
            self.space.step(self.timestep)
            time_count += self.timestep
            self.space.debug_draw(self.draw_options)
            
            pygame.display.flip()
            self.clock.tick(self.FPS)
            

    def run(self, act_list=None):
        if self.mode == 'play':
            self.play()
        elif self.mode == 'simulate':
            self.simulate(act_list)
        else:
            raise ValueError(f'No such mode {self.mode}. Mode list: (play, simulate, collect)')


if __name__ == '__main__':
    demo = IPHYRE()
    demo.run()

