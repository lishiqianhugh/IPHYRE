import sys
import pygame
from pygame.locals import *
import pymunk
import pymunk.pygame_util

from game_paras import game_paras

HEIGHT, WIDTH = 600, 600


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
    static_shape = pymunk.Segment(static_body, l_pos[0], l_pos[1], 10)
    static_shape.friction = friction
    static_shape.elasticity = elasticity
    if fix:
        static_shape.color = (0, 0, 255, 255)
    else:
        static_shape.color = (0, 255, 200, 255)
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
    num_ball = len(game_paras[game]['ball'])

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                sys.exit(0)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                p = event.pos
                eliminate(space, p, game_paras[game]['fix'], num_ball)

        space.step(1/60.0)
        screen.fill((255, 255, 255))
        space.debug_draw(draw_options)
        pygame.display.flip()
        clock.tick(50)


def collect_data():
    raise NotImplementedError()


if __name__ == '__main__':
    game, mode = sys.argv[1], sys.argv[2]
    if mode == 'play':
        simulate(game)
    elif mode == 'collect':
        collect_data()

