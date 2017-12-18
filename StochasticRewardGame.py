import numpy as np


class SRG:
    def __init__(self, sigma, sigma0, sigma1):
        self.sigma = sigma
        self.sigma0 = sigma0
        self.sigma1 = sigma1
        self.nb_actions = 3

        self.payoff_matrix = [
            [( 11, sigma0 ** 2), (-30, sigma  ** 2), (0, sigma ** 2)],
            [(-30, sigma  ** 2), (  7, sigma1 ** 2), (6, sigma ** 2)],
            [(  0, sigma  ** 2), (  0, sigma  ** 2), (5, sigma ** 2)]
        ]

    def play(self, i, j):
        mean, sigma = self.payoff_matrix[i][j]
        return np.random.normal(mean, sigma, 1)[0]
