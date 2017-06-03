#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 18:54:21 2017

@author: mortza
"""
from core import defs
import operator
import numpy as np
from numpy.random import randint, random
from core.GridWorld import GridWorld, Action, CellType


class Reward:

    def __init__(self):
        self.right = 0
        self.left = 0
        self.down = 0
        self.up = 0

    def __str__(self):
        ret = 'algorithms.qlearningbase.Reward(right={}, \
                left={}, down={}, up={})'
        return ret.format(self.right, self.left, self.down, self.up)


class QLearningBase:

    def __init__(self, reward_system, policy='e-greedy'):
        print('bootstrapping QLearningBase ...')
        self.alpha = defs.ALPHA
        self.gamma = defs.GAMMA
        self.epsilon = defs.EPSILON
        self.number_of_episodes = defs.NUMBER_OF_EPISODES
        # number of success
        self.number_of_success = 0
        # used to measure average number of moves for agent to go in goal
        self.number_of_moves = 0
        # maximum number of moves per episode
        self.maximum_number_of_moves = defs.MAXIMUM_NUMBER_OF_MOVES
        # a table for all cells in grid world with their 4 action
        self.q_table = [[Reward() for j in range(defs.NUMBER_OF_TILES_H)]
                        for i in range(defs.NUMBER_OF_TILES_V)]
        self.total_moves = np.zeros((self.number_of_episodes), dtype=np.int16)
        self.grid_world = GridWorld(reward_system=reward_system)
        if policy == 'soft_max':
            self.policy = self.soft_max_policy
        else:
            self.policy = self.e_greedy
        print('Done.')

    def e_greedy(self, p):
        # choose action with highest value
        action = Action.Right
        value = self.q_table[p.y][p.x].right

        actions = list([Action.Right, Action.Left, Action.Down, Action.Up])
        values = [self.q_table[p.y][p.x].right, self.q_table[p.y][
            p.x].left, self.q_table[p.y][p.x].down, self.q_table[p.y][p.x].up]
        for i in [0, 1, 2, 3]:
            if value < values[i]:
                value = values[i]
                action = actions[i]

        c_temp = self.grid_world.adjacent_of(p, action)
        if random() < (1 - self.epsilon) and \
                self.grid_world.cell_type_of(c_temp) != CellType.InvalidCell:
            # with probability of 1-epsilon choose best action
            return action
        else:
            # otherwise return a random action with equal probability
            st = set(actions)
            st.remove(action)
            actions = [a for a in st]

            a_temp = actions[randint(100) % 3]
            c_temp = self.grid_world.adjacent_of(p, a_temp)
            while self.grid_world.cell_type_of(c_temp) == CellType.InvalidCell:
                a_temp = actions[randint(100) % 3]
                c_temp = self.grid_world.adjacent_of(p, a_temp)
            return a_temp

    def soft_max(self, x):
        """
        """
        return np.exp(x) / np.sum(np.exp(x))

    def soft_max_policy(self, p):
        """
        """
        actions = list([Action.Right, Action.Left, Action.Down, Action.Up])
        values = np.array([self.q_table[p.y][p.x].right, self.q_table[p.y][
                          p.x].left, self.q_table[p.y][p.x].down,
            self.q_table[p.y][p.x].up])

        values = self.soft_max(values)

        pairs = dict(zip(actions, values))
        pairs = sorted(pairs.items(), key=operator.itemgetter(1))

        values = np.array([i[1] for i in pairs]).cumsum()
        actions = np.array([i[0] for i in pairs])

        rand_number = np.random.rand()

        if rand_number < values[0]:
            return actions[0]
        elif rand_number < values[1]:
            return actions[1]
        elif rand_number < values[2]:
            return actions[2]
        else:
            return actions[3]
