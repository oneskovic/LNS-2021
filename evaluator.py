import random
import sys
import os
import time

import pygame
from mpi4py import MPI

import subprocess
import pymunk
import pymunk.pygame_util
import pymunk.constraints
import numpy as np
from pymunk import Vec2d
from typing import List
from multiprocessing import cpu_count
from springed_body import SpringedBody

pymunk.pygame_util.positive_y_is_up = True
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class Evaluator:
    def __init__(self):
        pass

    def __make_ball(self, x, y, friction, mass):
        radius = 15
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        body = pymunk.Body(mass, inertia)
        body.position = x, y
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.friction = friction
        return shape, body


    def __construct_from_body(self, body: SpringedBody, space):
        nodes = []
        for node in body.nodes:
            x,y = body.node_positions[node]
            y += 10
            mass = body.node_mass[node]
            friction = body.node_friction[node]
            shape, body1 = self.__make_ball(x, y, friction, mass)
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


    def __contract_joints(self, joints, len):
        for joint in joints:
            joint.rest_length = len


    def __relax_joints(self, joints, len):
        for joint in joints:
            joint.rest_length = len


    def evaluate_with_display(self, body: SpringedBody):
        pygame.init()
        screen = pygame.display.set_mode((1500, 600))
        clock = pygame.time.Clock()
        running = True

        space = pymunk.Space()
        space.gravity = (0.0, -900.0)
        draw_options = pymunk.pygame_util.DrawOptions(screen)

        nodes, joints = self.__construct_from_body(body, space)

        ### walls
        static_lines = [
            pymunk.Segment(space.static_body, (-20000, 0.0), (20000.0, 0.0), 0.0),
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

        for t in range(1000):
            if remaining_times[current_state] <= 0:
                current_state = (current_state+1)%2
                if current_state == 0:
                    self.__contract_joints(joints, contracted_len)
                else:
                    self.__relax_joints(joints, extended_len)
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
            dt = 0.0001
            for x in range(100):
                space.step(dt)

            ### Flip screen
            pygame.display.flip()
            clock.tick(50)
            pygame.display.set_caption("fps: " + str(clock.get_fps()))
        pygame.quit()


    def evaluate_without_display(self, body: SpringedBody):
        space = pymunk.Space()
        space.gravity = (0.0, -900.0)

        nodes, joints = self.__construct_from_body(body, space)
        # Floor
        static_lines = [
            pymunk.Segment(space.static_body, (-20000, 0.0), (20000.0, 0.0), 0.0),
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
        for t in range(1000):
            if remaining_times[current_state] <= 0:
                current_state = (current_state+1)%2
                if current_state == 0:
                    self.__contract_joints(joints, contracted_len)
                else:
                    self.__relax_joints(joints, extended_len)
            if np.all(remaining_times <= 0):
                remaining_times = np.copy(state_times)
            remaining_times[current_state] -= 1

            dt = 0.001
            for i in range(10):
                space.step(dt)

        score = np.array([n.position.x for n in nodes]).min()
        return score


    def evaluate(self, body: SpringedBody, display=True):
        if display:
            return self.evaluate_with_display(body)
        else:
            return self.evaluate_without_display(body)


    def evaluate_organisms(self, organisms: List[SpringedBody]):
        worker_threads = int(os.getenv('THREAD_CNT')) - 1
        int_split = len(organisms) // worker_threads
        for i in range(worker_threads-1):
            sent_cnt = i*int_split
            organisms_to_send = organisms[sent_cnt:sent_cnt+int_split]
            comm.send(organisms_to_send, dest=i+1, tag=0)

        comm.send(organisms[(worker_threads-1)*int_split:], dest=worker_threads, tag=0)

        scores = np.array([])
        for i in range(worker_threads-1):
            data = np.zeros(int_split)
            comm.Recv([data, MPI.FLOAT], source=i+1, tag=1)
            scores = np.append(scores, data)

        data = np.zeros(len(organisms) - int_split*(worker_threads-1))
        comm.Recv([data, MPI.FLOAT], source=worker_threads, tag=1)
        scores = np.append(scores, data)

        for i in range(len(scores)):
            organisms[i].score = scores[i]
