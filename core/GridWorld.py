#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 16:51:53 2017

@author: mortza
"""
from enum import Enum
from . import defs
import numpy as np


class Action(Enum):
    Left = 1
    Right = 2
    Up = 3
    Down = 4


class CellType(Enum):
    # i.e. a cell taht is not goal and can agent go through
    Blank = 0
    # i.e. a cell that agent can'nt walk in
    Block = 1
    # i.e. goal cell
    Goal = 2
    # indicate invalid point
    InvalidCell = 3


class Point:

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __str__(self):
        return 'core.GridWorld.Point({}, {})'.format(self.x, self.y)


class GridWorld:

    def __init__(self):
        self.grid = np.zeros(
            (defs.NUMBER_OF_TILES_V, defs.NUMBER_OF_TILES_H), dtype=CellType)
        self.rewards = np.zeros(
            (defs.NUMBER_OF_TILES_V, defs.NUMBER_OF_TILES_H), dtype=np.int16)
        self.load_cell_from_file('./core/grid.txt')
        self.load_reward_from_file('./core/reward.txt')

    def opposite_of(self, action):
        """
        """
        if action == Action.Right:
            return Action.Left
        elif action == Action.Left:
            return Action.Right
        elif action == Action.Up:
            return Action.Down
        elif action == Action.Down:
            return Action.Up

    def actions_for(self, p):
        """
        returns a list of actions for input Point
        """
        ret = list()
        right = Point()
        left = Point()
        up = Point()
        down = Point()

        right.x = p.x + 1
        right.y = p.y

        left.x = p.x - 1
        left.y = p.y

        up.x = p.x
        up.y = p.y - 1

        down.x = p.x
        down.y = p.y + 1

        if self.can_move_to(right):
            ret.append(Action.Right)
        if self.can_move_to(left):
            ret.append(Action.Left)
        if self.can_move_to(up):
            ret.append(Action.Up)
        if self.can_move_to(down):
            ret.append(Action.Down)
        return ret

    def get_reward_of(self, p):
        """
        """
        if 0 <= p.x < defs.NUMBER_OF_TILES_H \
                and 0 <= p.y < defs.NUMBER_OF_TILES_V:
            return self.rewards[p.y][p.x]
        else:
            return -1

    def cell_type_of(self, p):
        """
        return cell type of input Point p
        """
        if 0 <= p.x < defs.NUMBER_OF_TILES_H \
                and 0 <= p.y < defs.NUMBER_OF_TILES_V:
            return self.grid[p.y][p.x]
        else:
            return CellType.InvalidCell

    def adjacent_of(self, p, a):
        """return adjacent point of `p` when move toward direction `a`
        """
        ret = Point()
        if a == Action.Right:
            ret.x = p.x + 1
            ret.y = p.y
        elif a == Action.Left:
            ret.x = p.x - 1
            ret.y = p.y
        elif a == Action.Up:
            ret.x = p.x
            ret.y = p.y - 1
        else:
            ret.x = p.x
            ret.y = p.y + 1
        return ret

    def can_move_to(self, p):
        """
        return true or false,
        p is point
        """
        if self.cell_type_of(p) == CellType.InvalidCell:
            return False
        elif self.cell_type_of(p) == CellType.Block:
            return False
        else:
            return True

    def load_cell_from_file(self, file_name):
        """
        read grid from file_name
        """
        with open(file_name, mode='r') as grid_file:
            for (i, line) in enumerate(grid_file.readlines()):
                for (j, char) in enumerate(line.split()):
                    self.grid[i][j] = CellType(int(char))

        if defs.SHOW_GRID_WORLD_VALUES:
            for i in range(defs.NUMBER_OF_TILES_V):
                for j in range(defs.NUMBER_OF_TILES_H):
                    print(self.grid[i][j], end=' ')
                print()

    def load_reward_from_file(self, file_name):
        """
        read rewards from file_name
        """
        with open(file_name, mode='r') as grid_file:
            for (i, line) in enumerate(grid_file.readlines()):
                for (j, char) in enumerate(line.split()):
                    self.rewards[i][j] = np.int16(char)

        if defs.SHOW_GRID_WORLD_VALUES:
            for i in range(defs.NUMBER_OF_TILES_V):
                for j in range(defs.NUMBER_OF_TILES_H):
                    print(self.rewards[i][j], end=' ')
                print()
