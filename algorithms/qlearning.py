#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 20:57:58 2017

@author: mortza
"""
from . import qlearningbase
from core.GridWorld import Action, CellType, Point
from core import defs
from numpy.random import randint


class StandardQLearning(qlearningbase.QLearningBase):

    def __init__(self):
        print("bootstrapping StandardQLearning ...")
        super(StandardQLearning, self).__init__()
        print("Done.")

    def train(self):
        print("Start StandardQLearning.train() execution")
        uid_h = defs.NUMBER_OF_TILES_H
        uid_v = defs.NUMBER_OF_TILES_V

        for episode in range(self.number_of_episodes):
            s = Point(x=randint(uid_h), y=randint(uid_v))
            while not self.grid_world.can_move_to(s) and \
                    self.grid_world.cell_type_of(s) != CellType.Goal:
                s.x = randint(uid_h)
                s.y = randint(uid_v)

            moves_per_episode = 0

            while self.grid_world.cell_type_of(s) != CellType.Goal and \
                    moves_per_episode < self.maximum_number_of_moves:
                a = self.policy(s)

                s_prime = self.grid_world.adjacent_of(s, a)

                r = self.grid_world.get_reward_of(s)

                actions_prime = self.grid_world.actions_for(s_prime)

                max_q_s_prime = self.Q(s_prime, actions_prime[0])

                for i in range(len(actions_prime)):
                    if self.Q(s_prime, actions_prime[i]) > max_q_s_prime:
                        max_q_s_prime = self.Q(s_prime, actions_prime[i])

                q_s_a = self.alpha * \
                    (r + self.gamma * max_q_s_prime - self.Q(s, a))

                self.update_qmatrix(s, a, q_s_a)

                if self.grid_world.cell_type_of(s_prime) != CellType.Block:
                    s.x = s_prime.x
                    s.y = s_prime.y
                moves_per_episode = moves_per_episode + 1

        if defs.LOG_ALGO_MSG:
            p = Point()
            for i in range(uid_h):
                for j in range(uid_v):
                    p.x = i
                    p.y = j
                    out = '({}, {}): {} {} {} {}'
                    print(out.format(i, j,
                                     self.Q(p, Action.Right),
                                     self.Q(p, Action.Left),
                                     self.Q(p, Action.Up),
                                     self.Q(p, Action.Down)))

        print('End StandardQLearning.train() execution')

    def Q(self, point, action):
        if self.grid_world.cell_type_of(point) != CellType.InvalidCell:
            if action == Action.Right:
                return self.q_table[point.y][point.x].right
            elif action == Action.Left:
                return self.q_table[point.y][point.x].left
            elif action == Action.Up:
                return self.q_table[point.y][point.x].up
            else:
                return self.q_table[point.y][point.x].down

    def R(self, point, action):
        """
                return Reward for a state (point) and action
        """
        t_p = self.grid_world.adjacent_of(point, action)
        if self.grid_world.cell_type_of(t_p) == CellType.Block:
            return -1
        elif self.grid_world.cell_type_of(t_p) == CellType.Goal:
            return -1
        else:
            return -1

    def update_qmatrix(self, point, action, new_value):
        if defs.LOG_ALGO_MSG:
            print('Update with: {}'.format(new_value))

        if action == Action.Right:
            self.q_table[point.y][point.x].right = self.q_table[
                point.y][point.x].right + new_value
        elif action == Action.Left:
            self.q_table[point.y][point.x].left = self.q_table[
                point.y][point.x].left + new_value
        elif action == Action.Up:
            self.q_table[point.y][point.x].up = self.q_table[
                point.y][point.x].up + new_value
        else:
            self.q_table[point.y][point.x].down = self.q_table[
                point.y][point.x].down + new_value


class OppositeQlearning(qlearningbase.QLearningBase):
    def __init__(sel, f):
        super(OppositeQlearning, self).__init__()
