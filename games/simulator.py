import sys
import time
import pygame
from pygame.locals import *
import pymunk
import pymunk.pygame_util
import pymunk.matplotlib_util
import numpy as np
import os
from copy import deepcopy
import sys
sys.path.append('D:\Files\Research\Projects\Interactive_Physical_Reasoning\IPHYRE')

from games.game_paras import game_paras
from games.game_paras import max_obj_num, max_eli_obj_num
from games.solutions import sol
from utils import write_json
import pdb
import cv2


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

    def __init__(self, game='support', fps=60):
        self.game = game
        self.HEIGHT, self.WIDTH = 600, 600
        self.FPS = fps
        self.timestep = 1 / self.FPS
        self.max_time = 15

        self.blocks = game_paras[self.game]['block']
        self.balls = game_paras[self.game]['ball']
        self.shape = [1] * len(self.blocks) + [0] * len(self.balls)
        self.num_ball = len(self.balls)
        self.eli = deepcopy(game_paras[self.game]['eli'])
        self.eli_mask = None
        self.dynamic = deepcopy(game_paras[self.game]['dynamic'])
        self.num_obj = len(self.eli)
        self.max_obj_num = max_obj_num
        self.max_eli_obj_num = max_eli_obj_num

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
        self.screen = None

        self.step_reward = -10 / self.FPS
        self.eli_reward = -100
        self.success_reward = 500

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
                self.eli_mask.pop(i)
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
    
    def get_property(self, body, idx, shape_flag):
        """
        For blocks:
            Given position and a, b
            Return the two end points, radius, eli, dynamic, joint and spring
        For balls:
            Return duplicated center position, radius, eli, dynamic, joint and spring
        """
        x, y = body.position
        shape = list(body.shapes)[0]
        if shape_flag:
            r = 10
            a_x, a_y = shape.a[0], shape.a[1]
            b_x, b_y = shape.b[0], shape.b[1]
            x1, x2 = x + a_x, x + b_x
            y1, y2 = y + a_y, y + b_y
            prop = [x1, y1, x2, y2, r, self.eli[idx], self.dynamic[idx], 0, 0]
        else:
            r = shape.radius
            prop = [x, y, x, y, r, self.eli[idx], self.dynamic[idx], 0, 0]
        if 'joint' in game_paras[self.game].keys():
            if idx in sum(game_paras[self.game]['joint'], []):
                prop[-2] = 1
        if 'spring' in game_paras[self.game].keys():
            if idx in sum(game_paras[self.game]['spring'], []):
                prop[-1] = 1
        return prop
    
    def get_all_property(self):
        all_property = np.zeros((self.max_obj_num, 9))
        for i, body in enumerate(self.space.bodies):
            index = self.eli_mask[i]
            all_property[index] = self.get_property(body, i, self.shape[i])
        return all_property

    def get_action_space(self):
        self.reset()
        actions = [[0., 0.]]  # no action
        all_property = self.get_all_property()
        for property in all_property:
            if property[-4] == 1:
                x = (property[0] + property[2]) / 2
                y = (property[1] + property[3]) / 2
                actions.append([x, y])
        for _ in range(self.max_eli_obj_num + 1 - len(actions)):
            actions.append([0., 0.])
        return actions

    def init_screen(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption(f"Interactive Physical Reasoning: {self.game}")
        self.button = self.Button(500, 500, 80, 20, 'Reset')
        self.start_button = self.Button(500, 450, 80, 20, 'Start', color='orange')
        self.clock = pygame.time.Clock()
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

    def reset(self, use_images=False):
        for body in self.space.bodies:
            shape = list(body.shapes)[0]
            self.space.remove(shape, shape.body)
        for joint in self.space.constraints:
            self.space.remove(joint)
        if self.screen:
            self.screen.fill((255, 255, 255))
        self.add_all()
        self.shape = [1] * len(self.blocks) + [0] * len(self.balls)
        self.eli = deepcopy(game_paras[self.game]['eli'])
        self.eli_mask = [i for i in range(len(self.space.bodies))]
        self.dynamic = deepcopy(game_paras[self.game]['dynamic'])

        if use_images:
            self.space.debug_draw(self.draw_options)
            pygame.display.flip()
            image = pygame.surfarray.array3d(self.screen)
            image = image.swapaxes(0,1)
            image[:,:,[0,2]] = image[:,:,[2,0]]
            # cv2.imshow('img', image)
            # cv2.waitKey(1000)
            return image
        else:
            return self.get_all_property()

    def step(self, pos, use_images=False):
        reward = self.step_reward
        done = False
        if pos == [0., 0.]:
            self.space.step(self.timestep)
        else:
            index = self.eliminate(pos)
            self.space.step(self.timestep)
            if index != -1:
                reward += self.eli_reward

        if self.examine_success():
            reward += self.success_reward
            done = True
        
        if use_images:
            self.screen.fill((255, 255, 255))
            self.space.debug_draw(self.draw_options)
            pygame.display.flip()
            image = pygame.surfarray.array3d(self.screen)
            image = image.swapaxes(0,1)
            image[:,:,[0,2]] = image[:,:,[2,0]]
            return image, reward, done
        else:
            return self.get_all_property(), reward, done

    def simulate(self, action=None):
        step, time_count = 0, 0
        total_step = len(action)
        while time_count < self.max_time:
            if step < total_step:
                p, t = action[step][0:2], action[step][2]
                if time_count >= t:
                    if self.eliminate(p) != -1:
                        step += 1

            self.space.step(self.timestep)
            time_count += self.timestep
            if self.examine_success():
                return 1, step, time_count
        return 0, step, time_count

    def simulate_vis(self, action=None):
        self.init_screen()
        self.reset()
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
     
    def play(self):
        self.init_screen()
        self.reset()
        finish_game, exceed_time, start = False, False, False
        time_count = 0
        while time_count < self.max_time + self.timestep:
            self.screen.fill((255, 255, 255))
            self.button_process()
            for event in pygame.event.get():
                if event.type == QUIT:
                    sys.exit(0)
                elif event.type == KEYDOWN and event.key == K_ESCAPE:
                    sys.exit(0)
                elif event.type == KEYDOWN and event.key == K_SPACE:
                    start = True
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    p = event.pos
                    button_pressed = self.button_process()
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

    def collect_initial_data(self, save_path='./dataset/game_initial_data/'):
        self.init_screen()
        self.reset()
        self.screen.fill((255, 255, 255))
        game_path = save_path + f'{self.game}/'
        if not os.path.exists(game_path):
            os.makedirs(game_path)
        dic = {'balls': np.array(self.balls),
               'blocks': np.array(self.blocks),
               'eli': np.array(self.eli),
               'dynamic': np.array(self.dynamic)}
        np.save(game_path + 'raw.npy', dic)
        vectors = self.get_all_property()
        np.save(game_path + 'vectors.npy', vectors)

        self.space.debug_draw(self.draw_options)
        pygame.display.flip()
        pygame.image.save(self.screen, game_path + f'{self.game}.jpg')
    
    def collect_data(self, save_path='./dataset/offline_data/', act_lists=None, fps=10):  # maximum fps=60
        # actions is a list of actions
        self.init_screen()
        game_path = save_path + f'{self.game}/'
        if not os.path.exists(game_path):
            os.makedirs(game_path)
        dic = {'balls': np.array(self.balls),
               'blocks': np.array(self.blocks),
               'eli': np.array(self.eli),
               'dynamic': np.array(self.dynamic)}
        np.save(game_path + 'raw.npy', dic)
        for i, act_list in enumerate(act_lists):  # the step number of each action can be variant
            self.reset()
            data_path = game_path + f'{i}/'
            if not os.path.exists(data_path):
                img_path = data_path + 'images/'
                os.makedirs(img_path)
                act_pos = np.array([list(a[0:-1]) for a in act_list])
                act_ts = np.array([a[-1] for a in act_list])
                np.save(data_path + 'actions.npy', np.concatenate((act_pos, act_ts.reshape(-1, 1)), axis=-1))
            else:
                continue  # already get the action_data stored
            step, time_count, save_count = 0, 0., 0
            total_step = len(act_list)
            interval = self.FPS / fps
            interval_cal = 0
            vectors = np.zeros((self.max_time * fps, self.max_obj_num, 9))

            while time_count < self.max_time:
                if step < total_step:
                    p, t = act_list[step][0:-1], act_list[step][-1]
                    if time_count >= t:
                        index = self.eliminate(p)
                        if index != -1:
                            print(f'Step {step}: Click {p} at time {time_count}.')
                            step += 1
                            for body in self.space.bodies:
                                print(body.position)

                if interval_cal == interval or interval_cal == 0:
                    interval_cal = 0
                    vectors[save_count] = self.get_all_property()
                    self.space.debug_draw(self.draw_options)
                    pygame.display.flip()
                    pygame.image.save(self.screen, img_path + f'{save_count}.jpg')
                    save_count += 1

                self.screen.fill((255, 255, 255))
                self.space.step(self.timestep)
                interval_cal += 1
                time_count += self.timestep
                if self.examine_success():
                    print(f'###### Success at time {time_count} ######')
                    for body in self.space.bodies:
                        print(body.position)
                    np.save(data_path + 'vectors.npy', np.array(vectors))
                    break

            np.save(data_path + 'vectors.npy', np.array(vectors))
    
    def collect_while_play(self, player_name='', max_episode=5, save_path='./dataset/player_data.json'):
        self.init_screen()
        self.reset()
        time_count, total_reward, actions = 0, 0, []
        status = 0
        reset = True
        exit = False
        episode = 0
        p = [0., 0.]
        while time_count < self.max_time:
            self.screen.fill((255, 255, 255))
            self.button_process()
            for event in pygame.event.get():
                if event.type == QUIT:
                    exit = True
                elif event.type == KEYDOWN and event.key == K_ESCAPE:
                    exit = True
                elif event.type == KEYDOWN and event.key == K_SPACE:
                    if reset and episode < max_episode:
                        status = 1
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    p = event.pos
                    button_pressed = self.button_process()
                    if button_pressed and episode < max_episode:
                        time_count, total_reward, actions = 0, 0, []
                        status = 0
                        reset = True
            if exit:
                break
            if status == 0:
                self.add_text(text=f"Press space to start!", loc=(150, 30), color="black")
            elif status == 1:
                _, reward, done = self.step(p, self.timestep)
                total_reward += reward
                if reward not in [self.step_reward, self.success_reward + self.step_reward]:
                    actions.append(list(p) + [time_count])

                time_count += self.timestep
                if time_count >= self.max_time - self.timestep:
                    info_dict = {"player": player_name, "game": self.game, "episode": episode, "actions": actions, "reward": total_reward}
                    write_json(save_path, info_dict)
                    status = 2
                    episode += 1
                    reset = False
                else:
                    if done:
                        info_dict = {"player": player_name, "game": self.game, "episode": episode, "actions": actions, "reward": total_reward}
                        write_json(save_path, info_dict)
                        status = 3
                        episode += 1
                        reset = False
            elif status == 2:
                self.add_text(text=f"Score in episode {episode} / {max_episode}: {round(total_reward, 2)}", 
                loc=(120, 30), color="red", font=40)
                if episode > max_episode - 1:
                    self.add_text(text=f"Press Esc to move on to the next game.",
                        loc=(120, 60), color="black", font=30)
            elif status == 3:
                self.add_text(text=f"Score in episode {episode} / {max_episode}: {round(total_reward, 2)}", 
                loc=(120, 30), color="green", font=40)
                if episode > max_episode - 1:
                    self.add_text(text=f"Press Esc to move on to the next game.",
                        loc=(120, 60), color="black", font=30)
            else:
                pass
            self.space.debug_draw(self.draw_options)
            pygame.display.flip()
            self.clock.tick(self.FPS)
        
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


def collect_play_all(player_name, max_episode, save_path):
    GAMES = list(game_paras.keys())[1:2]
    for game in GAMES:
        demo = IPHYRE(game)
        demo.collect_while_play(player_name=player_name, max_episode=max_episode, save_path=save_path)

if __name__ == '__main__':
    game = 'angle'
    mode = 'play'
    if mode == 'collect data':
        def return_center(p):
            x = (p[0][0] + p[1][0]) / 2
            y = (p[0][1] + p[1][1]) / 2
            return [x, y]
        def time_order(a):
            return a[-1]
        succeed_actions = np.load(f'dataset/action_data_7s/' + game + f'/succeed_actions_50.npy')
        fail_actions = np.load(f'dataset/action_data_7s/' + game + f'/fail_actions_50.npy')
        actions = np.concatenate((succeed_actions[:, :-3], fail_actions[:, :-3]))
        eli_idx = np.where(np.array(game_paras[game]['eli']) == 1)
        eli_blocks = np.array(game_paras[game]['block'])[eli_idx]
        act_lists = []
        for act_time in actions:
            act = [return_center(eli_blocks[i]) + [t] for i, t in enumerate(act_time) if t != 0]
            act.sort(key=time_order)
            act_lists.append(act)
        demo = IPHYRE(game)
        demo.collect_data(act_lists=act_lists)
    elif mode == 'play':
        demo = IPHYRE(game)
        demo.play()
    elif mode == 'collec_while_play':
        collect_play_all(player_name='lsq', save_path ='./dataset/player_data.json')
    else:
        raise ValueError(f'No such mode {mode}.')

