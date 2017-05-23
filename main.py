#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 17:00:39 2017

@author: mortza
"""

from core import GridWorld
from algorithms import qlearningbase
from core import defs


gw = GridWorld.GridWorld()
k = qlearningbase.QLearningBase()
for i in range(defs.NUMBER_OF_TILES_V):
    for j in range(defs.NUMBER_OF_TILES_H):
        print(k.q_table[i][j])
