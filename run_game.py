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
        self.game, self.mode = sys.argv[1], sys.argv[2]
        self.HEIGHT, self.WIDTH = 600, 600
        self.FPS = 60
        self.timestep = 1 / self.FPS
        self.max_time = 10

        self.blocks = game_paras[self.game]['block']
        self.balls = game_paras[self.game]['ball']
        self.shape = [1] * len(self.blocks) + [0] * len(self.balls)
        self.property = []

        self.num_ball = len(self.balls)
        self.eli = deepcopy(game_paras[self.game]['eli'])
        self.dynamic = deepcopy(game_paras[self.game]['dynamic'])

        self.b_mass, self.b_elasticity, self.b_friction = 1.0, 0.1, 0.5
        self.l_friction, self.l_elasticity = 0.5, 0.1
        self.G = (0., 100.0)

        self.space = pymunk.Space()
        self.space.gravity = self.G

        self.solutions = sol

        if self.mode != 'collect':
            pygame.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption(f"Interactive Physical Reasoning: {self.game}")
            self.clock = pygame.time.Clock()
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

    def reset(self):
        for body in self.space.bodies:
            shape = list(body.shapes)[0]
            self.space.remove(shape, shape.body)
        self.screen.fill((255, 255, 255))
        self.add_all()
        self.shape = [1] * len(self.blocks) + [0] * len(self.balls)
        self.eli = deepcopy(game_paras[self.game]['eli'])
        self.dynamic = deepcopy(game_paras[self.game]['dynamic'])

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

    def add_all(self):
        assert len(self.blocks) == len(game_paras[self.game]['eli'][:-self.num_ball])
        for l_para, eli, dynamics in zip(self.blocks, game_paras[self.game]['eli'][:-self.num_ball], game_paras[self.game]['dynamic'][:-self.num_ball]):
            if dynamics:
                self.add_dynamic_line(l_para, self.l_friction, self.l_elasticity)
            else:
                self.add_static_line(l_para, eli, self.l_friction, self.l_elasticity)
        for b_para in self.balls:
            self.add_ball(b_para[:2], b_para[2], self.b_mass, self.b_elasticity, self.b_friction)

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

    def add_text(self, text="Success!", loc=(230, 30), color="green", font=50):
        font = pygame.font.Font(None, font)
        text = font.render(text, True, pygame.Color(color))
        self.screen.blit(text, loc)

    def play(self):
        self.add_all()
        finish_game = False
        exceed_time = False
        time_count = 0
        while time_count < self.max_time + self.timestep:
            self.screen.fill((255, 255, 255))
            for event in pygame.event.get():
                if event.type == QUIT:
                    sys.exit(0)
                elif event.type == KEYDOWN and event.key == K_ESCAPE:
                    sys.exit(0)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    p = event.pos
                    self.eliminate(p)
                elif event.type == KEYDOWN and event.key == K_SPACE and finish_game:
                    finish_game = False
                    time_count = 0
                    self.reset() 
            time_count += self.timestep
            if time_count >= self.max_time - self.timestep:
                self.add_text(text="Failed", loc=(245, 30), color="red")
                self.add_text(text="Press space to Restart", loc=(200, 70), color="blue", font=30)
                finish_game = True
                time_count = self.max_time
                exceed_time = True
              
            if not exceed_time and self.examine_success():
                self.add_text(text="Success!", loc=(230, 30), color="green")
                self.add_text(text="Press space to Restart", loc=(200, 70), color="blue", font=30)
                finish_game = True
                time_count = 0             
            self.space.step(self.timestep)
            self.space.debug_draw(self.draw_options)
            
            pygame.display.flip()
            self.clock.tick(self.FPS)

    def simulate(self, action=None):
        self.add_all()
        step, time_count = 0, 0
        total_step = len(action)
        while time_count < self.max_time:
            self.screen.fill((255, 255, 255))
            if step < total_step:
                p, t = action[step][0], action[step][1]
                if time_count >= t:
                    # p = space.bodies[a].position
                    if self.eliminate(p) != -1:
                        print(f'Step {step}: Click {p} at time {time_count}.')
                        step += 1

            self.space.step(self.timestep)
            time_count += self.timestep
            self.space.debug_draw(self.draw_options)
            if self.examine_success():
                print(f'###### Success at time {time_count} ######')
                self.add_text()
                pygame.display.flip()
                time.sleep(2)
                sys.exit()
            pygame.display.flip()
            self.clock.tick(self.FPS)
        self.add_text(text="Failed", loc=(245, 30), color="red")
        pygame.display.flip()
        time.sleep(2)

    def get_property(self, body, shape_flag):
        '''
        For blocks:
            Given position and a,b; return the two points of block and radius
        For balls:
            return center position and radius,padding 2 zeros
        '''
        x, y = body.position
        shape = list(body.shapes)[0]
        if shape_flag:
            r = 10
            a_x, a_y = shape.a[0], shape.a[1]
            b_x, b_y = shape.b[0], shape.b[1]
            x1 = x + a_x
            x2 = x + b_x
            y1 = y + a_y
            y2 = y + b_y
            return [x1, y1, x2, y2, r]
        else:
            r = shape.radius
            return [x, y, 0, 0, r]

    def collect_data(self, save_path='data/', act_lists=None, fps=2):  # maximum fps=60
        # actions is a list of actions
        self.add_all()
        for i, act_list in enumerate(act_lists):  # the step number of each action can be variant
            eli_mask = np.arange(len(self.space.bodies))
            data_path = save_path + f'{self.game}/' + f'{i}/'
            if not os.path.exists(data_path):
                img_path = data_path + 'images/'
                os.makedirs(img_path)
                act_pos = np.array([list(a[0]) for a in act_list])
                act_ts = np.array([a[1] for a in act_list])
                np.save(data_path + 'actions.npy', np.concatenate((act_pos, act_ts.reshape(-1, 1)), axis=-1))
            else:
                continue  # already get the data stored
            step, time_count = 0, 0.
            total_step = len(act_list)
            interval = self.FPS / fps
            interval_cal = 0
            vectors = [[]] * len(self.space.bodies)

            while time_count < self.max_time:
                if step < total_step:
                    p, t = act_list[step][0], act_list[step][1]
                    if time_count >= t:
                        # p = space.bodies[a].position
                        index = self.eliminate(p)
                        if index != -1:
                            print(f'Step {step}: Click {p} at time {time_count}.')
                            step += 1
                            eli_mask = np.delete(eli_mask, index)
                            for body in self.space.bodies:
                                print(body.position)

                self.space.step(self.timestep)
                if interval_cal == interval or interval_cal == 0:
                    interval_cal = 0
                    print(f'{round(time_count, 1)}: eli_mask:{eli_mask}')
                    for i, body in enumerate(self.space.bodies):
                        shape = [self.shape[i]]
                        property = self.get_property(body, shape[0])
                        eli = [self.eli[i]]
                        dynamic = [self.dynamic[i]]
                        index = eli_mask[i]
                        vectors[index].append(shape + property + eli + dynamic)  # 1 + 5 + 1 + 1
                    for j in range(len(vectors)):
                        if j not in eli_mask:
                            vectors[j].append([0] * len(vectors[0][0]))
                    print(f'number of bodies:{len(self.space.bodies)}')
                    # self.draw_options = pymunk.SpaceDebugDrawOptions()
                    fig = plt.figure(figsize=(10, 10))
                    ax = plt.axes(xlim=(0, self.HEIGHT), ylim=(0, self.WIDTH))
                    # ax = plt.axes()
                    ax.set_aspect("equal")
                    ax.set_axis_off()
                    ax.invert_yaxis()
                    o = pymunk.matplotlib_util.DrawOptions(ax)
                    self.space.debug_draw(o)
                    fig.savefig(img_path + f'{round(time_count, 1)}.jpg')
                interval_cal += 1
                time_count += self.timestep
                if self.examine_success():
                    print(f'###### Success at time {time_count} ######')
                    for body in self.space.bodies:
                        print(body.position)
                    np.save(data_path + 'vectors.npy', np.array(vectors))
                    sys.exit()

            np.save(data_path + 'vectors.npy', np.array(vectors))
            sys.exit()

    def run(self, act_list=None):
        act_lists = []
        if act_list is None:
            act_list = self.solutions[self.game]
            act_lists.append(act_list)
        if self.mode == 'play':
            self.play()
        elif self.mode == 'simulate':
            self.simulate(act_list)
        elif self.mode == 'collect':
            self.collect_data(act_lists=act_lists)
        else:
            raise ValueError(f'No such mode {self.mode}. Mode list: (play, simulate, collect)')


if __name__ == '__main__':
    demo = IPHYRE()
    demo.run()

