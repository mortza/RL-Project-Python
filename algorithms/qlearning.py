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

    def __init__(self, reward_system='file'):
        print("bootstrapping StandardQLearning ...")
        super(StandardQLearning, self).__init__(reward_system)
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

                actions_prime = self.grid_world.actions_for(s_prime)

                max_q_s_prime = self.Q(s_prime, actions_prime[0])

                for i in range(len(actions_prime)):
                    if self.Q(s_prime, actions_prime[i]) > max_q_s_prime:
                        max_q_s_prime = self.Q(s_prime, actions_prime[i])

                if self.grid_world.reward_type == 2:
                    r_s_a = self.grid_world.get_reward_of(s_prime)
                elif self.grid_world.reward_type == 1:
                    r_s_a = self.grid_world.get_reward_of(s_prime, s)

                q_s_a = self.Q(s, a)
                q_s_a = q_s_a + self.alpha * \
                    (r_s_a + self.gamma * max_q_s_prime - q_s_a)

                self.update_qmatrix(s, a, q_s_a)

                if self.grid_world.cell_type_of(s_prime) != CellType.Block:
                    s.x = s_prime.x
                    s.y = s_prime.y
                moves_per_episode = moves_per_episode + 1

            self.total_moves[episode] = moves_per_episode

        if defs.LOG_ALGO_MSG:
            self._log()

        print('End PaperQLearning.train() execution')

    def _log(self):
        uid_h = defs.NUMBER_OF_TILES_H
        uid_v = defs.NUMBER_OF_TILES_V
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
            self.q_table[point.y][point.x].right = new_value
        elif action == Action.Left:
            self.q_table[point.y][point.x].left = new_value
        elif action == Action.Up:
            self.q_table[point.y][point.x].up = new_value
        else:
            self.q_table[point.y][point.x].down = new_value


class PaperQLearning(qlearningbase.QLearningBase):

    def __init__(self, reward_system='file'):
        print("bootstrapping PaperQLearning ...")
        super(PaperQLearning, self).__init__(reward_system)
        print("Done.")

    def train(self):
        print("Start PaperQLearning.train() execution")
        uid_h = defs.NUMBER_OF_TILES_H
        uid_v = defs.NUMBER_OF_TILES_V

        for episode in range(self.number_of_episodes):
            # Initialize s
            s = Point(x=randint(uid_h), y=randint(uid_v))
            while not self.grid_world.can_move_to(s) and \
                    self.grid_world.cell_type_of(s) != CellType.Goal:
                s.x = randint(uid_h)
                s.y = randint(uid_v)

            moves_per_episode = 0

            while self.grid_world.cell_type_of(s) != CellType.Goal and \
                    moves_per_episode < self.maximum_number_of_moves:
                # Choose an action a, from state s using policy derived from Q
                a = self.policy(s)
                # Take action, observe reward r and next state s'
                s_prime = self.grid_world.adjacent_of(s, a)

                # Determine opposite action (ã)
                a_tilde = self.grid_world.opposite_of(a)

                s_prime_actions = self.grid_world.actions_for(s_prime)
                a_star = s_prime_actions[0]
                for a_t in s_prime_actions:
                    if self.Q(s_prime, a_star) < self.Q(s_prime, a_t):
                        a_star = a_t

                a_tilde_star = s_prime_actions[0]
                for a_t in s_prime_actions:
                    if self.Q(s_prime, a_tilde_star) > self.Q(s_prime, a_t):
                        a_tilde_star = a_t

                if self.grid_world.reward_type == 2:
                    r_s_a = self.grid_world.get_reward_of(s_prime)
                elif self.grid_world.reward_type == 1:
                    r_s_a = self.grid_world.get_reward_of(s_prime, s)

                q_s_a = self.Q(s, a)
                q_s_a_tilde = self.Q(s, a_tilde)
                q_s_prime_a_star = self.Q(s_prime, a_star)
                q_s_prime_a_tilde_star = self.Q(s_prime, a_tilde_star)

                temp_state = self.grid_world.adjacent_of(s, a_tilde)
                if self.grid_world.reward_type == 2:
                    r_s_a_tilde = self.grid_world.get_reward_of(temp_state)
                elif self.grid_world.reward_type == 1:
                    r_s_a_tilde = self.grid_world.get_reward_of(temp_state, s)

                if q_s_a < q_s_prime_a_star:
                    temp1 = q_s_a + self.alpha * \
                        (r_s_a + self.gamma * q_s_prime_a_star +
                         (1 - self.gamma) * q_s_prime_a_tilde_star - q_s_a)

                    temp2 = q_s_a_tilde + self.alpha * \
                        (r_s_a_tilde + self.gamma * q_s_prime_a_tilde_star +
                         (1 - self.gamma) * q_s_prime_a_star - q_s_a_tilde)

                    self.update_qmatrix(s, a, temp1)
                    self.update_qmatrix(s, a_tilde, temp2)
                else:
                    temp1 = q_s_a + self.alpha * \
                        (r_s_a + (1 - self.gamma) * q_s_prime_a_star +
                         self.gamma * q_s_prime_a_tilde_star - q_s_a)

                    temp2 = q_s_a_tilde + self.alpha * \
                        (r_s_a_tilde + (1 - self.gamma) *
                         q_s_prime_a_tilde_star + self.gamma *
                            q_s_prime_a_star - q_s_a_tilde)

                    self.update_qmatrix(s, a, temp1)
                    self.update_qmatrix(s, a_tilde, temp2)
                if self.grid_world.cell_type_of(s_prime) != CellType.Block:
                    s.x = s_prime.x
                    s.y = s_prime.y
                moves_per_episode = moves_per_episode + 1

            self.total_moves[episode] = moves_per_episode
        if defs.LOG_ALGO_MSG:
            self._log()

        print('End PaperQLearning.train() execution')

    def _log(self):
        uid_h = defs.NUMBER_OF_TILES_H
        uid_v = defs.NUMBER_OF_TILES_V
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
            self.q_table[point.y][point.x].right = new_value
        elif action == Action.Left:
            self.q_table[point.y][point.x].left = new_value
        elif action == Action.Up:
            self.q_table[point.y][point.x].up = new_value
        else:
            self.q_table[point.y][point.x].down = new_value
