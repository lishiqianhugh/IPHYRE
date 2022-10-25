import sys
import pygame
from pygame.locals import *
import pymunk
import pymunk.pygame_util

from game_paras import game_paras

game, mode = sys.argv[1], sys.argv[2]
HEIGHT, WIDTH = 600, 600
FPS = 60
timestep = 1 / FPS
max_time = 10
num_ball = len(game_paras[game]['ball'])
b_mass, b_elasticity, b_friction = 1.0, 0.1, 0.5
l_friction, l_elasticity = 0.5, 1.0
G = (0., 100.0)


def add_ball(space, b_pos, radius, mass, elasticity, friction):
    moment = pymunk.moment_for_circle(mass, 0, radius)
    body = pymunk.Body(mass, moment)
    body.position = b_pos[0], b_pos[1]
    shape = pymunk.Circle(body, radius)
    shape.elasticity = elasticity
    shape.friction = friction
    shape.color = (255, 0, 0, 255)
    space.add(body, shape)
    return shape


def add_line(space, l_pos, fix, friction, elasticity):
    static_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    x1, y1, x2, y2 = l_pos[0][0], l_pos[0][1], l_pos[1][0], l_pos[1][1]
    x, y = (x1 + x2) / 2, (y1 + y2) / 2
    static_body.position = x, y
    static_shape = pymunk.Segment(static_body, (x1-x, y1-y), (x2-x, y2-y), 10)
    # static_shape = pymunk.Segment(static_body, l_pos[0], l_pos[1], 10)
    static_shape.friction = friction
    static_shape.elasticity = elasticity
    if fix:
        static_shape.color = (0, 0, 0, 255)
    else:
        static_shape.color = (164, 164, 164, 255)
    space.add(static_body, static_shape)
    return static_shape


def eliminate(space, p, fix):
    for i, body in enumerate(space.bodies[:-num_ball]):
        shape = list(body.shapes)[0]
        x, y = body.position
        min_0 = min(shape.a[0], shape.b[0]) + x - 10
        max_0 = max(shape.a[0], shape.b[0]) + x + 10
        min_1 = min(shape.a[1], shape.b[1]) + y - 10
        max_1 = max(shape.a[1], shape.b[1]) + y + 10
        if fix[i] == 0 and min_0 < p[0] < max_0 and min_1 < p[1] < max_1:
            space.remove(shape, shape.body)
            return True
    return False


def examine_success(space):
    success = 0
    for ball in space.bodies[-num_ball:]:
        if ball.position[1] > HEIGHT:
            success += 1
    if success == num_ball:
        return True
    else:
        return False


def add_text(screen):
    font = pygame.font.Font(None, 100)
    text = "Success!"
    text = font.render(text, True, pygame.Color("green"))
    screen.blit(text, (150, 100))


def play(game='support', gravity=G):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Interactive Physical Reasoning: {game}")
    clock = pygame.time.Clock()

    space = pymunk.Space()
    space.gravity = gravity

    draw_options = pymunk.pygame_util.DrawOptions(screen)
    # add bodies
    assert len(game_paras[game]['block']) == len(game_paras[game]['fix'])
    for l_para, fix in zip(game_paras[game]['block'], game_paras[game]['fix']):
        add_line(space, l_para, fix, l_friction, l_elasticity)
    for b_para in game_paras[game]['ball']:
        add_ball(space, b_para[:2], b_para[2], b_mass, b_elasticity, b_friction)

    while True:
        screen.fill((255, 255, 255))
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                sys.exit(0)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                p = event.pos
                eliminate(space, p, game_paras[game]['fix'])

        space.step(timestep)
        space.debug_draw(draw_options)
        if examine_success(space):
            add_text(screen)
        pygame.display.flip()
        clock.tick(FPS)


def simulate(game='support', action=None, gravity=G):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Interactive Physical Reasoning: {game}")
    clock = pygame.time.Clock()

    space = pymunk.Space()
    space.gravity = gravity

    draw_options = pymunk.pygame_util.DrawOptions(screen)
    # add bodies
    assert len(game_paras[game]['block']) == len(game_paras[game]['fix'])
    for l_para, fix in zip(game_paras[game]['block'], game_paras[game]['fix']):
        add_line(space, l_para, fix, l_friction, l_elasticity)
    for b_para in game_paras[game]['ball']:
        add_ball(space, b_para[:2], b_para[2], b_mass, b_elasticity, b_friction)

    step, time = 0, 0
    total_step = len(action)
    while time < max_time:
        screen.fill((255, 255, 255))
        if step < total_step:
            p, t = action[step][0], action[step][1]
            if time >= t:
                # p = space.bodies[a].position
                if eliminate(space, p, game_paras[game]['fix']):
                    print(f'Step {step}: Click {p} at time {time}.')
                    step += 1

        space.step(timestep)
        time += timestep
        space.debug_draw(draw_options)
        if examine_success(space):
            add_text(screen)
        pygame.display.flip()
        clock.tick(FPS)


def collect_data(game='support', action=None, gravity=G):
    space = pymunk.Space()
    space.gravity = gravity
    # add bodies
    assert len(game_paras[game]['block']) == len(game_paras[game]['fix'])
    for l_para, fix in zip(game_paras[game]['block'], game_paras[game]['fix']):
        add_line(space, l_para, fix, l_friction, l_elasticity)
    for b_para in game_paras[game]['ball']:
        add_ball(space, b_para[:2], b_para[2], b_mass, b_elasticity, b_friction)

    step, time = 0, 0
    total_step = len(action)
    while time < max_time:
        if step < total_step:
            p, t = action[step][0], action[step][1]
            if time >= t:
                # p = space.bodies[a].position
                if eliminate(space, p, game_paras[game]['fix']):
                    print(f'Step {step}: Click {p} at time {time}.')
                    step += 1
                    for body in space.bodies:
                        print(body.position)

        space.step(timestep)
        time += timestep
        if examine_success(space):
            print(f'###### Success at time {time} ######')
            for body in space.bodies:
                print(body.position)
            sys.exit()


if __name__ == '__main__':
    if mode == 'play':
        play(game)
    elif mode == 'simulate':
        action = [[(450., 320.), 1], [(500., 320.), 1.5]]  # hinder
        simulate(game, action)
    elif mode == 'collect':
        action = [[(450., 320.), 1], [(500., 320.), 1.5]]  # hinder
        collect_data(game, action)
    else:
        raise ValueError(f'No such mode {mode}. Mode list: (play, simulate, collect)')
