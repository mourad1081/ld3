import numpy as np
import plots
from MultiAgent import MultiAgent
from StochasticRewardGame import SRG


def second_exercise():
    nb_steps = 5000
    nb_agents = 5000
    kk = range(nb_agents)
    game1 = SRG(sigma=0.2, sigma0=0.2, sigma1=0.2)
    game2 = SRG(sigma=0.1, sigma0=4.0, sigma1=0.1)
    game3 = SRG(sigma=0.1, sigma0=4.0, sigma1=4.0)
    multi_agents_softmax_1 = [MultiAgent('softmax', game1, nb_steps) for _ in kk]
    multi_agents_softmax_2 = [MultiAgent('softmax', game2, nb_steps) for _ in kk]
    multi_agents_softmax_3 = [MultiAgent('softmax', game3, nb_steps) for _ in kk]
    multi_agents_heuristic_1 = [MultiAgent('heuristic', game1, nb_steps) for _ in kk]
    multi_agents_heuristic_2 = [MultiAgent('heuristic', game2, nb_steps) for _ in kk]
    multi_agents_heuristic_3 = [MultiAgent('heuristic', game3, nb_steps) for _ in kk]

    # learning phase
    for a in kk:
        multi_agents_softmax_1[a].learn()
        multi_agents_softmax_2[a].learn()
        multi_agents_softmax_3[a].learn()
        print('multi agent soft', a, 'finished learning !')
        multi_agents_heuristic_1[a].learn()
        multi_agents_heuristic_2[a].learn()
        multi_agents_heuristic_3[a].learn()
        print('multi agent heur', a, 'finished learning !')

    avg_s_g1 = []
    avg_s_g2 = []
    avg_s_g3 = []

    avg_h_g1 = []
    avg_h_g2 = []
    avg_h_g3 = []

    for s in range(nb_steps):
        sum_avg_s_g1 = 0.0
        sum_avg_h_g1 = 0.0

        sum_avg_s_g2 = 0.0
        sum_avg_h_g2 = 0.0

        sum_avg_h_g3 = 0.0
        sum_avg_s_g3 = 0.0

        for a in kk:
            sum_avg_s_g1 += multi_agents_softmax_1[a].get_stats()[1][s]
            sum_avg_h_g1 += multi_agents_heuristic_1[a].get_stats()[1][s]

            sum_avg_s_g2 += multi_agents_softmax_2[a].get_stats()[1][s]
            sum_avg_h_g2 += multi_agents_heuristic_2[a].get_stats()[1][s]

            sum_avg_s_g3 += multi_agents_softmax_3[a].get_stats()[1][s]
            sum_avg_h_g3 += multi_agents_heuristic_3[a].get_stats()[1][s]

        avg_s_g1.append(sum_avg_s_g1 / nb_agents)
        avg_h_g1.append(sum_avg_h_g1 / nb_agents)

        avg_s_g2.append(sum_avg_s_g2 / nb_agents)
        avg_h_g2.append(sum_avg_h_g2 / nb_agents)

        avg_s_g3.append(sum_avg_s_g3 / nb_agents)
        avg_h_g3.append(sum_avg_h_g3 / nb_agents)

    plots.multiplot([range(nb_steps), range(nb_steps)],
                    [avg_s_g1, avg_h_g1],
                    'Total average reward for both players '
                    'σ = 0.2, σ_0 = 0.2, σ_1 = 0.2',
                    'Step number',
                    'Average reward',
                    ['Softmax with τ = 0.1', 'FMQ'])

    plots.multiplot([range(nb_steps), range(nb_steps)],
                    [avg_s_g2, avg_h_g2],
                    'Total average reward for both players '
                    'σ = 0.1, σ_0 = 4.0, σ_1 = 0.1',
                    'Step number',
                    'Average reward',
                    ['Softmax with τ = 0.1', 'FMQ'])

    plots.multiplot([range(nb_steps), range(nb_steps)],
                    [avg_s_g3, avg_h_g3],
                    'Total average reward for both players '
                    'σ = 0.1, σ_0 = 4.0, σ_1 = 4.0',
                    'Step number',
                    'Average reward',
                    ['Softmax with τ = 0.1', 'FMQ'])
