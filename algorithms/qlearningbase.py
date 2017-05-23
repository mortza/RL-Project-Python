#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 18:54:21 2017

@author: mortza
"""
from core import defs
from numpy.random import randint, random
from core.GridWorld import GridWorld, Action


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

    def __init__(self):
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
        self.grid_world = GridWorld()
        print('Done.')

    def policy(self, p):
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

        if random() < (1 - self.epsilon):
            # with probability of 1-epsilon choose best action
            return action
        else:
            # otherwise return a random action with equal probability
            st = set(actions)
            st.remove(action)
            actions = [a for a in st]
            return actions[randint(100) % 3]
