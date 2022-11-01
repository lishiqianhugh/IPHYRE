import sys
import time
import pygame
from pygame.locals import *
import pymunk
import pymunk.pygame_util
import matplotlib as mpl
import matplotlib.pyplot as plt
import pymunk.matplotlib_util

from game_paras import game_paras
import pdb

class IPHYRE():
    def __init__(self):
        self.game, self.mode = sys.argv[1], sys.argv[2]
        self.HEIGHT, self.WIDTH = 600, 600
        self.FPS = 60
        self.timestep = 1 / self.FPS
        self.max_time = 10

        self.blocks = game_paras[self.game]['block']
        self.eli = game_paras[self.game]['eli']
        self.dynamic = game_paras[self.game]['dynamic']
        self.balls = game_paras[self.game]['ball']
        self.num_ball = len(self.balls)
        self.b_mass, self.b_elasticity, self.b_friction = 1.0, 0.1, 0.5
        self.l_friction, self.l_elasticity = 0.5, 0.1
        self.G = (0., 100.0)

        self.space = pymunk.Space()
        self.space.gravity = self.G

        self.solutions = {
            'support': [[(250., 400.), 1], [(250., 500.), 1.5]],
            'hinder': [[(450., 320.), 1], [(500., 320.), 1.5]],
            'direction': [[(150., 180.), 1], [(100., 350.), 1.5]],
            'hole': [[(250., 100.), 1], [(250., 150.), 2.]],
            'multi_balls': [[(500., 400.), 1]],
            'fill': []
        }

        if self.mode != 'collect':
            pygame.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption(f"Interactive Physical Reasoning: {self.game}")
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
        return shape

    def add_static_line(self, l_pos, eli, friction, elasticity):
        static_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        x1, y1, x2, y2 = l_pos[0][0], l_pos[0][1], l_pos[1][0], l_pos[1][1]
        x, y = (x1 + x2) / 2, (y1 + y2) / 2
        static_body.position = x, y
        static_shape = pymunk.Segment(static_body, (x1-x, y1-y), (x2-x, y2-y), 10)
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
        shape = pymunk.Segment(body, (x1-x, y1-y), (x2-x, y2-y), 10.)
        shape.friction = friction
        shape.elasticity = elasticity
        self.space.add(body, shape)

    def add_all(self):
        assert len(self.blocks) == len(self.eli)
        for l_para, eli, dynamics in zip(self.blocks, self.eli, self.dynamic):
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
                return True
        return False

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

    def play(self):
        self.add_all()
        while True:
            self.screen.fill((255, 255, 255))
            for event in pygame.event.get():
                if event.type == QUIT:
                    sys.exit(0)
                elif event.type == KEYDOWN and event.key == K_ESCAPE:
                    sys.exit(0)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    p = event.pos
                    self.eliminate(p)

            self.space.step(self.timestep)
            self.space.debug_draw(self.draw_options)
            if self.examine_success():
                self.add_text()
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
                    if self.eliminate(p):
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

    def collect_data(self, action=None,fps=2): #maxmimum fps=60
        self.add_all()
        step, time_count = 0, 0
        total_step = len(action)
        interval  =  self.FPS / fps
        interval_cal = 0
        positions = []
        while time_count < self.max_time:
            if step < total_step:
                p, t = action[step][0], action[step][1]
                if time_count >= t:
                    # p = space.bodies[a].position
                    if self.eliminate(p):
                        print(f'Step {step}: Click {p} at time {time_count}.')
                        step += 1
                        for body in self.space.bodies:
                            print(body.position)

            self.space.step(self.timestep)
            if interval_cal == interval or interval_cal ==0:
                interval_cal = 0
                for body in self.space.bodies:
                    positions.append(body.position)
            #self.draw_options = pymunk.SpaceDebugDrawOptions()
                fig = plt.figure(figsize=(14,10))
                ax = plt.axes(xlim=(0, 1000), ylim=(0, 1000))
                ax.set_aspect("equal")
                ax.invert_yaxis()
                o = pymunk.matplotlib_util.DrawOptions(ax)
                a = self.space.debug_draw(o)
                fig.savefig('try.png')
                pdb.set_trace()

            interval_cal += 1
            time_count += self.timestep

            if self.examine_success():
                print(f'###### Success at time {time_count} ######')
                for body in self.space.bodies:
                    print(body.position)
                sys.exit()

    def run(self, action=None):
        if action is None:
            action = self.solutions[self.game]
        if self.mode == 'play':
            self.play()
        elif self.mode == 'simulate':
            self.simulate(action)
        elif self.mode == 'collect':
            self.collect_data(action)
        else:
            raise ValueError(f'No such mode {self.mode}. Mode list: (play, simulate, collect)')


if __name__ == '__main__':
    demo = IPHYRE()
    demo.run()
