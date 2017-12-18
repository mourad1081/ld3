import random

import numpy as np
from math import exp, sqrt


class MultiAgent:
    def __init__(self, ai, game, nb_steps, tau=0.1):
        self.ai = ai
        self.game = game
        self.policies = {
            'softmax': self.softmax,
            'heuristic': self.heuristic
        }

        self.rewards = [[0], [0], [0]]
        self.rewards_agent_1 = [[0], [0], [0]]
        self.rewards_agent_2 = [[0], [0], [0]]

        self.total_reward_average = [0]
        self.total_reward = 0.0
        self.tau = tau
        self.nb_steps = nb_steps
        self.agent_1_turn = True

    def softmax(self, exponent_function=None):
        f = self.Q if exponent_function is None else exponent_function
        boltzmann_distribution = self.get_boltzmann_distribution(exponent_function=f)
        action = generate_random(distribution=boltzmann_distribution)
        return action

    def heuristic(self):
        return self.softmax(self.FMQ)

    def Q(self, a):
        if self.agent_1_turn:
            weight = len(self.rewards_agent_2[a]) / (sum([len(self.rewards_agent_2[i]) for i in range(3)]))
            mean = np.mean(self.rewards_agent_1[a])
        else:
            weight = len(self.rewards_agent_1[a]) / (sum([len(self.rewards_agent_1[i]) for i in range(3)]))
            mean = np.mean(self.rewards_agent_2[a])
        return mean * weight

    def FMQ(self, a):
        q_a, e_a, r_max_a, f_a = self.Q(a), None, None, None
        if self.agent_1_turn:
            r_max_a = np.max(self.rewards_agent_1[a])
            f_a = 1 / len(self.rewards_agent_1)
        else:
            r_max_a = np.max(self.rewards_agent_2[a])
            f_a = 1 / len(self.rewards_agent_2)
        return (1 - f_a) * q_a + f_a * r_max_a

    def random_choice(self):
        return random.randint(0, self.game.nb_actions - 1)

    def learn(self):
        for i in range(self.nb_steps):
            self.agent_1_turn = True
            action_agent_1 = self.policies[self.ai]()
            self.agent_1_turn = False
            action_agent_2 = self.policies[self.ai]()

            reward = self.game.play(action_agent_1, action_agent_2)
            self.update_rewards(action_agent_1, action_agent_2, reward, i)
            # update tau
            if self.ai == 'heuristic':
                tau_start = 100
                tau_end = 0.1
                decay_factor = 0.1
                self.tau = exp(-decay_factor * i) * tau_start + tau_end

    def update_rewards(self, action_agent_1, action_agent_2, reward, i):
        self.total_reward += reward
        self.rewards_agent_1[action_agent_1].append(reward)
        self.rewards_agent_2[action_agent_2].append(reward)
        self.total_reward_average.append(self.total_reward / (i + 1))

    def get_boltzmann_distribution(self, exponent_function):
        distribution = []
        for a in range(self.game.nb_actions):
            denominator = 0.0
            for b in range(self.game.nb_actions):
                denominator += exp(exponent_function(b) / self.tau)
            distribution.append(exp(exponent_function(a) / self.tau) / denominator)
        return distribution

    def get_stats(self):
        return range(self.nb_steps), self.total_reward_average


def generate_random(distribution):
    return np.random.choice([0, 1, 2], p=distribution)
