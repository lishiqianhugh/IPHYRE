import sys
import pygame
from pygame.locals import *
import pymunk
import pymunk.pygame_util

from game_paras import game_paras

HEIGHT, WIDTH = 600, 600
timestep = 1 / 60.0
game, mode = sys.argv[1], sys.argv[2]
num_ball = len(game_paras[game]['ball'])


def add_ball(space, b_pos, radius, mass, elasticity):
    moment = pymunk.moment_for_circle(mass, 0, radius)
    body = pymunk.Body(mass, moment)
    body.position = b_pos[0], b_pos[1]
    shape = pymunk.Circle(body, radius)
    shape.elasticity = elasticity
    shape.color = (255, 0, 0, 255)
    space.add(body, shape)
    return shape


def add_line(space, l_pos=((360.0, 360.0), (360.0, 600.0)), fix=0, friction=0.4, elasticity=1.0,):
    static_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    # static_body.position = l_pos[0][0] + l_pos[1][0] / 2, l_pos[0][1] + l_pos[1][1] / 2
    static_shape = pymunk.Segment(static_body, l_pos[0], l_pos[1], 10)
    static_shape.friction = friction
    static_shape.elasticity = elasticity
    if fix:
        static_shape.color = (0, 0, 0, 255)
    else:
        static_shape.color = (164, 164, 164, 255)
    space.add(static_body, static_shape)
    return static_shape


def add_wall(space):
    static_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    static_wall = [pymunk.Segment(static_body, (0, 0), (0, HEIGHT), 10),
                   pymunk.Segment(static_body, (0, 0), (WIDTH, 0), 10),
                   pymunk.Segment(static_body, (WIDTH, HEIGHT), (WIDTH, 0), 10),
                   pymunk.Segment(static_body, (WIDTH, HEIGHT), (0, HEIGHT), 10)]
    space.add(static_body)
    for line in static_wall:
        line.elasticity = 0.9
        line.friction = 0.4
        space.add(line)
    return static_wall


def eliminate(space, p, fix, num_ball):
    for i, body in enumerate(space.bodies[:-num_ball]):
        shape = list(body.shapes)[0]
        min_0 = min(shape.a[0], shape.b[0]) - 10
        max_0 = max(shape.a[0], shape.b[0]) + 10
        min_1 = min(shape.a[1], shape.b[1]) - 10
        max_1 = max(shape.a[1], shape.b[1]) + 10
        if fix[i] == 0 and min_0 < p[0] < max_0 and min_1 < p[1] < max_1:
            space.remove(shape, shape.body)


def examine_success(space, num_ball):
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


def simulate(game='support', gravity=(0., 100.0)):
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
        add_line(space, l_para, fix)
    for b_para in game_paras[game]['ball']:
        add_ball(space, b_para[:2], b_para[2], mass=1.0, elasticity=0.1)

    while True:
        screen.fill((255, 255, 255))
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                sys.exit(0)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                p = event.pos
                eliminate(space, p, game_paras[game]['fix'], num_ball)

        space.step(timestep)
        space.debug_draw(draw_options)
        if examine_success(space, num_ball):
            add_text(screen)
        pygame.display.flip()
        clock.tick(50)


def collect_data(game='support', action=None, gravity=(0., 100.0)):
    space = pymunk.Space()
    space.gravity = gravity
    # add bodies
    assert len(game_paras[game]['block']) == len(game_paras[game]['fix'])
    for l_para, fix in zip(game_paras[game]['block'], game_paras[game]['fix']):
        add_line(space, l_para, fix)
    for b_para in game_paras[game]['ball']:
        add_ball(space, b_para[:2], b_para[2], mass=1.0, elasticity=0.1)

    step, clock = 0, 0
    a, t = action[step][0], action[step][1]
    total_step = len(action)
    while True:
        for i, body in enumerate(space.bodies):
            print(body.position)
        if clock == t and step < total_step:
            p = space.bodies[a].position
            eliminate(space, p, game_paras[game]['fix'], num_ball)
        space.step(timestep)
        clock += timestep
        step += 1


if __name__ == '__main__':
    if mode == 'play':
        simulate(game)
    elif mode == 'collect':
        action = [[1, 1], [2, 2]]
        collect_data(game, action)
