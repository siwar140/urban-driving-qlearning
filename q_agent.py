# -*- coding: utf-8 -*-
"""Agent Q-Learning tabular avec epsilon-greedy décroissant."""
import numpy as np, random


class QLearningAgent:
    def __init__(self, num_states, num_actions=5,
                 alpha=0.15, gamma=0.95, epsilon=1.0, q_init=1.0):
        self.alpha       = alpha
        self.gamma       = gamma
        self.epsilon     = epsilon
        self.num_actions = num_actions
        # Initialisation légèrement optimiste pour favoriser l'exploration
        self.Q = np.full((num_states, num_actions), float(q_init))

    def choose_action(self, state_idx, is_training=True):
        if is_training and random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        q_row = self.Q[state_idx]
        best  = np.flatnonzero(q_row >= q_row.max() - 1e-9)
        return int(random.choice(best))

    def learn(self, s, a, r, s_next, terminal):
        target = r if terminal else r + self.gamma * self.Q[s_next].max()
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def save(self, path="q_table.npy"):
        np.save(path, self.Q)
        print(f"Q-table sauvegardee -> {path}  shape={self.Q.shape}")

    def load(self, path="q_table.npy"):
        self.Q = np.load(path)
        print(f"Q-table chargee <- {path}  shape={self.Q.shape}")