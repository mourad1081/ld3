import random
from math import exp, sqrt

import numpy


class Agent:

    def __init__(self, ai, game, nb_steps, epsilon=-1, tau=-1, is_exo_3=False):
        if ai == 'e-greedy' and epsilon == -1:
            raise ValueError("Tu ne peux pas choisir ε-greedy si tu ne spécifie pas un ε >= 0 !")
        if ai == 'softmax' and tau == -1:
            raise ValueError("Tu ne peux pas choisir softmax si tu ne spécifie pas un τ >= 0 !")
        self.ai = ai
        self.game = game
        self.is_exo_3 = is_exo_3
        self.policies = {
            'random': self.random_choice,
            'e-greedy': self.e_greedy,
            'softmax': self.softmax
        }

        self.rewards = [[] for _ in range(self.game.nb_actions)]

        self.total_reward = 0.0
        self.total_reward_average = []
        self.total_selected_actions = [0, 0, 0, 0]
        self.average_over_time = [[], [], [], []]  # liste 2D de mean over time
        self.tau = tau
        self.epsilon = epsilon
        if self.is_exo_3:
            self.epsilon = 1
            self.tau = 4*(999/1000)

        self.nb_steps = nb_steps

    def random_choice(self):
        return random.randint(0, self.game.nb_actions - 1)

    def e_greedy(self):
        return self.random_choice() if random.random() <= self.epsilon else self.best_action()

    def softmax(self):
        boltzmann_distribution = self.get_boltzmann_distribution(exponent_function=self.Q)
        action = generate_random(distribution=boltzmann_distribution)
        return action

    def learn(self):
        for i in range(self.nb_steps):
            if self.is_exo_3:
                if self.ai == 'softmax':
                    self.tau = 4.0 * ((1000.0 - i) / 1000.0)
                elif self.ai == 'e-greedy':
                    self.epsilon = 1.0 / sqrt(i + 1.0)

            action = self.policies[self.ai]()
            reward = self.game.get_reward(action)
            self.update_rewards(action, reward, i)
            self.total_selected_actions[action] += 1

    def update_rewards(self, action, reward, current_nb_steps):
        self.total_reward += reward
        self.rewards[action].append(reward)
        self.total_reward_average.append(self.total_reward / (current_nb_steps + 1))
        self.average_over_time[0].append(self.Q(0))
        self.average_over_time[1].append(self.Q(1))
        self.average_over_time[2].append(self.Q(2))
        self.average_over_time[3].append(self.Q(3))

    def Q(self, a):
        return numpy.mean(self.rewards[a]) if len(self.rewards[a]) > 0 else 0.0

    def get_boltzmann_distribution(self, exponent_function):
        distribution = []
        for a in range(self.game.nb_actions):
            denominator = 0.0
            for b in range(self.game.nb_actions):
                denominator += exp(exponent_function(b) / self.tau)
            distribution.append(exp(exponent_function(a) / self.tau) / denominator)
        return distribution

    def best_action(self):
        avg = -1
        index = self.random_choice()
        for i in range(self.game.nb_actions):
            if len(self.rewards[i]) > 0:
                m = self.Q(i)
                if m > avg:
                    avg, index = m, i
        return index

    def get_stats(self):
        return range(self.nb_steps), self.total_reward_average, self.total_selected_actions, self.average_over_time


def generate_random(distribution):
    return numpy.random.choice([0, 1, 2, 3], p=distribution)
