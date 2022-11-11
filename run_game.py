import sys
import time
import pygame
from pygame.locals import *
import pymunk
import pymunk.pygame_util
import matplotlib.pyplot as plt
import pymunk.matplotlib_util
import numpy as np
import pdb
import os
from copy import deepcopy

from game_paras import game_paras
from solutions import sol


class IPHYRE():
    class Button():
        def __init__(self, x, y, width, height, buttonText='Button', color='blue'):
            self.x = x
            self.y = y
            self.width = width
            self.height = height
            self.alreadyPressed = False
            self.font = pygame.font.SysFont('Arial', 20)
            self.fillColors = {
                'normal': '#ffffff',
                'hover': '#666666',
                'pressed': '#333333',
            }
            self.buttonSurface = pygame.Surface((self.width, self.height))
            self.buttonRect = pygame.Rect(self.x, self.y, self.width, self.height)

            self.buttonSurf = self.font.render(buttonText, True, color)

    def __init__(self, game='support', mode='play'):
        self.game, self.mode = game, mode
        self.HEIGHT, self.WIDTH = 600, 600
        self.FPS = 60
        self.timestep = 1 / self.FPS
        self.max_time = 15

        self.blocks = game_paras[self.game]['block']
        self.balls = game_paras[self.game]['ball']
        self.shape = [1] * len(self.blocks) + [0] * len(self.balls)
        self.property = []
        self.num_ball = len(self.balls)
        self.eli = deepcopy(game_paras[self.game]['eli'])
        self.dynamic = deepcopy(game_paras[self.game]['dynamic'])
        self.num_obj = len(self.eli)

        self.joint = None
        if 'joint' in game_paras[self.game].keys():
            self.joint = game_paras[self.game]['joint']
        self.spring = None
        if 'spring' in game_paras[self.game].keys():
            self.spring = game_paras[self.game]['spring']

        self.b_mass, self.b_elasticity, self.b_friction = 1.0, 0.1, 0.5
        self.l_friction, self.l_elasticity = 0.5, 0.1
        self.space = pymunk.Space()
        self.space.gravity = (0., 100.0)
        self.solutions = sol

        if self.mode not in ['collect', 'simulate']:
            pygame.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption(f"Interactive Physical Reasoning: {self.game}")
            self.button = self.Button(500, 500, 80, 20, 'Reset')
            self.start_button = self.Button(500, 450, 80, 20, 'Start', color='orange')
            self.clock = pygame.time.Clock()
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

    def add_ball(self, b_pos, radius, mass, elasticity, friction):
        moment = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, moment)
        body.position = b_pos[0], b_pos[1]
        shape = pymunk.Circle(body, radius)
        shape.elasticity = elasticity
        shape.friction = friction
        shape.color = (255, 0, 0, 255)
        self.space.add(body, shape)
        return body

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
        return static_body

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
        return body

    def add_joint(self):
        for (b1, b2) in game_paras[self.game]['joint']:
            c: pymunk.Constraint = pymunk.PinJoint(self.space.bodies[b1], self.space.bodies[b2])
            self.space.add(c)

    def add_spring(self):
        for (b1, b2) in game_paras[self.game]['spring']:
            c = pymunk.DampedSpring(self.space.bodies[b1], self.space.bodies[b2], (0, 0), (0, 0), 20, 1, 0.3)
            self.space.add(c)

    def add_all(self):
        assert len(self.blocks) == len(game_paras[self.game]['eli'][:-self.num_ball])
        for l_para, eli, dynamics in zip(self.blocks, game_paras[self.game]['eli'][:-self.num_ball],
                                         game_paras[self.game]['dynamic'][:-self.num_ball]):
            if dynamics:
                self.add_dynamic_line(l_para, self.l_friction, self.l_elasticity)
            else:
                self.add_static_line(l_para, eli, self.l_friction, self.l_elasticity)
        for b_para in self.balls:
            self.add_ball(b_para[:2], b_para[2], self.b_mass, self.b_elasticity, self.b_friction)
        if self.joint:
            self.add_joint()
        if self.spring:
            self.add_spring()

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
                for constraint in list(shape.body.constraints):
                    self.space.remove(constraint)
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

    def play(self):
        self.add_all()
        finish_game, exceed_time, start = False, False, False
        time_count = 0
        while time_count < self.max_time + self.timestep:
            self.screen.fill((255, 255, 255))
            self.button_process()
            self.start_process()
            for event in pygame.event.get():
                if event.type == QUIT:
                    sys.exit(0)
                elif event.type == KEYDOWN and event.key == K_ESCAPE:
                    sys.exit(0)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    p = event.pos
                    button_pressed = self.button_process()
                    start_pressed = self.start_process()
                    if start_pressed:
                        start = True
                    if button_pressed:
                        time_count = 0
                        finish_game, exceed_time, start = False, False, False
                    if not finish_game and start:
                        self.eliminate(p)
            if start:
                time_count += self.timestep
                if time_count >= self.max_time - self.timestep:
                    self.add_text(text="Failed", loc=(245, 30), color="red")
                    time_count = self.max_time
                    exceed_time, finish_game = True, True

                if not exceed_time and self.examine_success():
                    self.add_text(text="Success!", loc=(230, 30), color="green")
                    time_count = 0
                    finish_game = True
                self.space.step(self.timestep)
                self.space.debug_draw(self.draw_options)

                pygame.display.flip()
                self.clock.tick(self.FPS)
            else:
                self.space.debug_draw(self.draw_options)
                pygame.display.flip()

    def simulate(self, action=None):
        self.add_all()
        step, time_count = 0, 0
        total_step = len(action)
        while time_count < self.max_time:
            if step < total_step:
                p, t = action[step][0: 2], action[step][2]
                if time_count >= t:
                    if self.eliminate(p) != -1:
                        step += 1

            self.space.step(self.timestep)
            time_count += self.timestep
            if self.examine_success():
                return True
        return False

    def simulate_vis(self, action=None):
        self.add_all()
        step, time_count = 0, 0
        total_step = len(action)
        while time_count < self.max_time:
            self.screen.fill((255, 255, 255))
            if step < total_step:
                p, t = action[step][0: 2], action[step][2]
                if time_count >= t:
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
                break
            pygame.display.flip()
            self.clock.tick(self.FPS)
        self.add_text(text="Failed", loc=(245, 30), color="red")
        pygame.display.flip()
        time.sleep(2)

    def collect_while_play(self, save_path='data_player/', fps=2):
        self.add_all()
        game_path = save_path + f'{self.game}/'
        if not os.path.exists(game_path):
            os.makedirs(game_path)
        dic = {'balls': np.array(self.balls),
               'blocks': np.array(self.blocks),
               'eli': np.array(self.eli),
               'dynamic': np.array(self.dynamic)}
        np.save(game_path + 'property.npy', dic)
        finish_game, exceed_time, start = False, False, False
        time_count = 0

        eli_mask = np.arange(len(self.space.bodies))
        interval = 1 / fps
        actions = []
        while time_count < self.max_time + self.timestep:
            self.screen.fill((255, 255, 255))
            self.button_process()
            self.start_process()
            for event in pygame.event.get():
                if event.type == QUIT:
                    sys.exit(0)
                elif event.type == KEYDOWN and event.key == K_ESCAPE:
                    sys.exit(0)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    p = event.pos
                    button_pressed = self.button_process()
                    start_pressed = self.start_process()
                    if start_pressed:
                        start = True
                    if button_pressed:
                        time_count = 0
                        finish_game, exceed_time, start = False, False, False
                        actions = []
                        eli_mask = np.arange(len(self.space.bodies))
                    if not finish_game and start:
                        index = self.eliminate(p)
                        if index != -1:
                            eli_mask = np.delete(eli_mask, index)
                            actions.append(np.array(list(p) + [time_count]))

            if start:
                time_count += self.timestep
                if time_count >= self.max_time - self.timestep:
                    self.add_text(text="Failed", loc=(245, 30), color="red")
                    time_count = self.max_time
                    exceed_time = True
                    if not finish_game:
                        finish_game = True
                        num_dirs = 0
                        data_path = game_path + 'fail/'
                        if os.path.exists(data_path):
                            num_dirs = len(os.listdir(data_path))
                        os.makedirs(data_path + f'{num_dirs}/')
                        np.save(data_path + f'{num_dirs}/' + 'actions.npy', np.array(actions))

                if not exceed_time and self.examine_success():
                    self.add_text(text="Success!", loc=(230, 30), color="green")
                    time_count = 0
                    if not finish_game:
                        finish_game = True
                        num_dirs = 0
                        data_path = game_path + 'succeed/'
                        if os.path.exists(data_path):
                            num_dirs = len(os.listdir(data_path))
                        os.makedirs(data_path + f'{num_dirs}/')
                        np.save(data_path + f'{num_dirs}/' + 'actions.npy', np.array(actions))
                self.space.step(self.timestep)
                self.space.debug_draw(self.draw_options)

                pygame.display.flip()
                self.clock.tick(self.FPS)
            else:
                self.space.debug_draw(self.draw_options)
                pygame.display.flip()

    def collect_data(self, save_path='data/', act_lists=None, fps=2):  # maximum fps=60
        # actions is a list of actions
        self.add_all()
        game_path = save_path + f'{self.game}/'
        if not os.path.exists(game_path):
            os.makedirs(game_path)
        dic = {'balls': np.array(self.balls),
               'blocks': np.array(self.blocks),
               'eli': np.array(self.eli),
               'dynamic': np.array(self.dynamic)}
        np.save(game_path + 'property.npy', dic)
        for i, act_list in enumerate(act_lists):  # the step number of each action can be variant
            eli_mask = np.arange(len(self.space.bodies))
            data_path = game_path + f'{i}/'
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
                        index = eli_mask[i]
                        property = self.get_property(body, self.shape[i])
                        if self.joint:
                            if index in self.joint:
                                property[-2] = 1
                        if self.spring:
                            if index in self.spring:
                                property[-1] = 1
                        vectors[index].append(property)
                    for j in range(len(vectors)):
                        if j not in eli_mask:
                            vectors[j].append([0] * len(vectors[0][0]))
                    print(f'number of bodies:{len(self.space.bodies)}')
                    # self.draw_options = pymunk.SpaceDebugDrawOptions()
                    fig = plt.figure(figsize=(10, 10))
                    ax = plt.axes(xlim=(0, self.HEIGHT), ylim=(0, self.WIDTH))
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
                    break

            np.save(data_path + 'vectors.npy', np.array(vectors))

    def get_property(self, body, shape_flag):
        '''
        For blocks:
            Given position and a,b; return the two points of block and radius
        For balls:
            return center position and radius,padding 2 zeros
        the last two digits are left for joint and spring
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
            return [x1, y1, x2, y2, r, 0, 0]
        else:
            r = shape.radius
            return [x, y, 0, 0, r, 0, 0]

    def reset(self):
        for body in self.space.bodies:
            shape = list(body.shapes)[0]
            self.space.remove(shape, shape.body)
        for joint in self.space.constraints:
            self.space.remove(joint)
        self.screen.fill((255, 255, 255))
        self.add_all()
        self.shape = [1] * len(self.blocks) + [0] * len(self.balls)
        self.eli = deepcopy(game_paras[self.game]['eli'])
        self.dynamic = deepcopy(game_paras[self.game]['dynamic'])

    def button_process(self):
        mousePos = pygame.mouse.get_pos()
        self.button.buttonSurface.fill(self.button.fillColors['normal'])
        if self.button.buttonRect.collidepoint(mousePos):
            self.button.buttonSurface.fill(self.button.fillColors['hover'])
            if pygame.mouse.get_pressed(num_buttons=3)[0]:
                self.button.buttonSurface.fill(self.button.fillColors['pressed'])
                if not self.button.alreadyPressed:
                    self.reset()
                    self.button.alreadyPressed = True
            else:
                self.button.alreadyPressed = False
        self.button.buttonSurface.blit(self.button.buttonSurf, [
            self.button.buttonRect.width / 2 - self.button.buttonSurf.get_rect().width / 2,
            self.button.buttonRect.height / 2 - self.button.buttonSurf.get_rect().height / 2
        ])
        self.screen.blit(self.button.buttonSurface, self.button.buttonRect)
        return self.button.alreadyPressed

    def start_process(self):
        mousePos = pygame.mouse.get_pos()
        self.start_button.buttonSurface.fill(self.start_button.fillColors['normal'])
        if self.start_button.buttonRect.collidepoint(mousePos):
            self.start_button.buttonSurface.fill(self.start_button.fillColors['hover'])
            if pygame.mouse.get_pressed(num_buttons=3)[0]:
                self.start_button.buttonSurface.fill(self.start_button.fillColors['pressed'])
                if not self.start_button.alreadyPressed:
                    self.start_button.alreadyPressed = True
            else:
                self.start_button.alreadyPressed = False
        self.start_button.buttonSurface.blit(self.start_button.buttonSurf, [
            self.start_button.buttonRect.width / 2 - self.start_button.buttonSurf.get_rect().width / 2,
            self.start_button.buttonRect.height / 2 - self.start_button.buttonSurf.get_rect().height / 2
        ])
        self.screen.blit(self.start_button.buttonSurface, self.start_button.buttonRect)
        return self.start_button.alreadyPressed

    def add_text(self, text="Success!", loc=(230, 30), color="green", font=50):
        font = pygame.font.Font(None, font)
        text = font.render(text, True, pygame.Color(color))
        self.screen.blit(text, loc)

    def run(self, act_list=None):
        act_lists = []
        if act_list is None:
            act_list = self.solutions[self.game]
            act_lists.append(act_list)
        if self.mode == 'play':
            self.play()
        elif self.mode == 'simulate':
            return self.simulate(act_list)
        elif self.mode == 'simulate_vis':
            self.simulate_vis(act_list)
        elif self.mode == 'collect':
            self.collect_data(act_lists=act_lists)
        elif self.mode == 'collect_while_play':
            self.collect_while_play()
        else:
            raise ValueError(f'No such mode {self.mode}. '
                             f'Mode list: (play, simulate, simulate_vis, collect, collect_while_play)')
        return -1


if __name__ == '__main__':
    g, m = sys.argv[1], sys.argv[2]
    demo = IPHYRE(g, m)
    demo.run()
