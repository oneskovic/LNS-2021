import random
import sys

import pygame

import pymunk
import pymunk.pygame_util
import pymunk.constraints
import numpy as np
from pymunk import Vec2d
from springed_body import Organism

pymunk.pygame_util.positive_y_is_up = True

def make_ball(x, y, friction, mass):
    radius = 15
    inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
    body = pymunk.Body(mass, inertia)
    body.position = x, y
    shape = pymunk.Circle(body, radius, (0, 0))
    shape.friction = friction
    return shape, body

def construct_from_body(body: Organism, space):
    nodes = []
    for node in body.nodes:
        x,y = body.node_positions[node]
        y += 10
        mass = body.node_mass[node]
        friction = body.node_friction[node]
        shape, body1 = make_ball(x, y, friction, mass)
        space.add(body1, shape)
        nodes.append(body1)

    joints = []
    for key in body.connected_nodes.keys():
        spring_id = body.connected_nodes[key]
        damping = body.springs_damping[spring_id]
        node1 = key[0]
        node2 = key[1]
        joint = pymunk.constraints.DampedSpring(nodes[node1], nodes[node2], (0,0), (0, 0), 100, 80, damping)
        space.add(joint)
        joints.append(joint)

    return nodes, joints

def contract_joints(joints, len):
    for joint in joints:
        joint.rest_length = len

def relax_joints(joints, len):
    for joint in joints:
        joint.rest_length = len

def evaluate_with_display(body: Organism, max_steps):
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    clock = pygame.time.Clock()

    space = pymunk.Space()
    space.gravity = (0.0, -900.0)
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    nodes, joints = construct_from_body(body, space)

    ### walls
    static_lines = [
        pymunk.Segment(space.static_body, (-1000, 0.0), (1000.0, 0.0), 0.0),
    ]
    for l in static_lines:
        l.friction = 0.5
    space.add(*static_lines)

    ch = space.add_collision_handler(0, 0)
    ch.data["surface"] = screen

    total_period = 10
    contract_time = int(body.contract_time * total_period)
    contracted_len = 60
    extended_len = 140

    state_times = np.array([contract_time, total_period - contract_time], dtype=int)
    remaining_times = np.array([contract_time, total_period - contract_time], dtype=int)
    current_state = 0

    for t in range(max_steps):
        if remaining_times[current_state] <= 0:
            current_state = (current_state+1)%2
            if current_state == 0:
                contract_joints(joints, contracted_len)
            else:
                relax_joints(joints, extended_len)
        if np.all(remaining_times <= 0):
            remaining_times = np.copy(state_times)
        remaining_times[current_state] -= 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                pass
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                pass
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                pygame.image.save(screen, "contact_with_friction.png")

        ### Clear screen
        screen.fill(pygame.Color("white"))

        ### Draw stuff
        space.debug_draw(draw_options)

        ### Update physics
        dt = 0.01
        for x in range(1):
            space.step(dt)

        ### Flip screen
        pygame.display.flip()
        clock.tick(50)
        pygame.display.set_caption("fps: " + str(clock.get_fps()))
    pygame.quit()
    score = np.array([n.position.x for n in nodes]).min()
    return score

def evaluate_without_display(body: Organism, max_steps):
    space = pymunk.Space()
    space.gravity = (0.0, -900.0)

    nodes, joints = construct_from_body(body, space)
    # Floor
    static_lines = [
        pymunk.Segment(space.static_body, (-10000, 0.0), (10000.0, 0.0), 0.0),
    ]
    for l in static_lines:
        l.friction = 0.5
    space.add(*static_lines)

    ch = space.add_collision_handler(0, 0)

    total_period = 10
    contract_time = int(body.contract_time*total_period)
    contracted_len = 60
    extended_len = 140

    state_times = np.array([contract_time, total_period-contract_time], dtype=int)
    remaining_times = np.array([contract_time, total_period-contract_time], dtype=int)
    current_state = 0
    for t in range(max_steps):
        if remaining_times[current_state] <= 0:
            current_state = (current_state+1)%2
            if current_state == 0:
                contract_joints(joints, contracted_len)
            else:
                relax_joints(joints, extended_len)
        if np.all(remaining_times <= 0):
            remaining_times = np.copy(state_times)
        remaining_times[current_state] -= 1

        dt = 0.01
        space.step(dt)

    score = np.array([n.position.x for n in nodes]).min()
    return score


def evaluate(body: Organism, display=True, max_steps=1000):
    if display:
        return evaluate_with_display(body,max_steps)
    else:
        return evaluate_without_display(body, max_steps)

if __name__ == '__main__':

    body = Organism()
    body.add_node()
    body.add_node()
    body.add_node()
    body.add_node()
    for _ in range(100):
        body.make_random_connection()
    #body.make_random_connection()

    evaluate(body, True)