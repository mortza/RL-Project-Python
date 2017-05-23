#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 17:00:39 2017

@author: mortza
"""

from core import GridWorld
from algorithms import qlearning


gw = GridWorld.GridWorld()
k = qlearning.StandardQLearning()
k.train()
