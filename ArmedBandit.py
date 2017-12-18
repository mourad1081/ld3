import random
import numpy as np


class ArmedBandit:

    def __init__(self, nb_actions, factor_standard_deviation=1):
        self.nb_actions = nb_actions
        self.actions = {
            'q_ai*': [1.3, 1.1, 0.5, 0.3],
            'sigma_i': [factor_standard_deviation * 0.9,
                        factor_standard_deviation * 0.6,
                        factor_standard_deviation * 0.4,
                        factor_standard_deviation * 2]
        }
        # self.reward_total = 0

    def get_reward(self, i):
        mean, sigma = self.actions['q_ai*'][i], self.actions['sigma_i'][i]
        return np.random.normal(mean, sigma, 1)[0]

    def select_random_action(self):
        return random.randint(0, self.nb_actions - 1)

    def update_rewards(self):
        pass

