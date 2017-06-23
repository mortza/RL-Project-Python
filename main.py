#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 17:00:39 2017

@author: mortza
"""
from algorithms import qlearning
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from core.GridWorld import Point, CellType
from core import defs


matplotlib.style.use('ggplot')

pql = qlearning.PaperQLearning(reward_system='dist', policy='soft_max')
pql.train()

sql = qlearning.StandardQLearning(reward_system='dist', policy='soft_max')
sql.train()

steps_array_pql = []
steps_array_sql = []
rng = np.arange(100) + 1

for k in range(100):
    uid_h = defs.NUMBER_OF_TILES_H
    uid_v = defs.NUMBER_OF_TILES_V

    s = Point(x=np.random.randint(uid_h),
              y=np.random.randint(uid_v))
    while not pql.grid_world.can_move_to(s, ignore_block=False) and \
            pql.grid_world.cell_type_of(s) != CellType.Goal:
        s.x = np.random.randint(uid_h)
        s.y = np.random.randint(uid_v)
    steps_array_pql.append(pql.find_goal(s))
    steps_array_sql.append(sql.find_goal(s))

plt.plot(rng, steps_array_pql, label='PQL')
plt.plot(rng, steps_array_sql, label='SQL')
plt.legend(loc='upper right')
plt.show()
print(np.mean(steps_array_pql))
print(np.mean(steps_array_sql))
