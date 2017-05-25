#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 17:00:39 2017

@author: mortza
"""
from algorithms import qlearning
import matplotlib.pyplot as plt

pql = qlearning.PaperQLearning()
pql.train()

sql = qlearning.StandardQLearning()
sql.train()


plt.plot(range(len(pql.total_moves)), pql.total_moves,
         'r--', label="PaperQLearning")
plt.plot(range(len(pql.total_moves)), sql.total_moves,
         'b--', label="StandardQLearning")
plt.legend()
plt.show()
