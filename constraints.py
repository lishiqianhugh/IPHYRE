"""A L shape attached with a joint and constrained to not tip over.

This example is also used in the Get Started Tutorial. 
"""

__docformat__ = "reStructuredText"

import random
import sys

import pygame

import pymunk
import pymunk.pygame_util

random.seed(1)


def add_ball(space):
    """Add a ball to the given space at a random position"""
    mass = 1
    radius = 14
    inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
    body = pymunk.Body(mass, inertia)
    x = random.randint(120, 380)
    body.position = x, 50
    shape = pymunk.Circle(body, radius, (0, 0))
    shape.friction = 1
    space.add(body, shape)
    return shape


def add_L(space):
    """Add a inverted L shape with two joints"""
    rotation_center_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    rotation_center_body.position = (300, 400)

    # rotation_limit_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    # rotation_limit_body.position = (200, 300)

    body = pymunk.Body(10, 10000)
    body.position = (300, 300)
    # l1 = pymunk.Segment(body, (-145, 0), (255.0, 0.0), 1)
    # l2 = pymunk.Segment(body, (-145, 0), (-145.0, -25.0), 1)
    # l1.friction = 1
    # l2.friction = 1
    rotation_center_joint = pymunk.PinJoint(body, rotation_center_body, (0, 0), (0, 0))
    # joint_limit = 25
    # rotation_limit_joint = pymunk.SlideJoint(
    #     body, rotation_limit_body, (-100, 0), (0, 0), 0, joint_limit
    # )
    #space.add(rotation_center_body)
    space.add(body, rotation_center_joint)
    #space.add(l1, l2, body, rotation_center_joint, rotation_limit_joint)
    return True


def main():
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    pygame.display.set_caption("Joints. Just wait and the L will tip over")
    clock = pygame.time.Clock()

    space = pymunk.Space()
    space.gravity = (0.0, 900.0)

    lines = add_L(space)
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                sys.exit(0)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                pygame.image.save(screen, "slide_and_pinjoint.png")


        space.step(1 / 50.0)

        screen.fill((255, 255, 255))
        space.debug_draw(draw_options)

        pygame.display.flip()
        clock.tick(50)


if __name__ == "__main__":
    main()
