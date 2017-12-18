import plots
from Agent import Agent
from ArmedBandit import ArmedBandit


def first_exercise():
    nb_actions = 4
    nb_steps = 1000
    nb_agents = 1000
    zero_to_nb_agents = range(nb_agents)
    game = ArmedBandit(nb_actions, factor_standard_deviation=1)

    agents_random = [Agent('random', game, nb_steps, is_exo_3=False) for _ in zero_to_nb_agents]
    agents_softmax_1 = [Agent('softmax', game, nb_steps, is_exo_3=False, tau=1) for _ in zero_to_nb_agents]
    agents_softmax_01 = [Agent('softmax', game, nb_steps, is_exo_3=False, tau=0.1) for _ in zero_to_nb_agents]
    agents_egreedy_e00 = [Agent('e-greedy', game, nb_steps, is_exo_3=False, epsilon=0.0) for _ in zero_to_nb_agents]
    agents_egreedy_e01 = [Agent('e-greedy', game, nb_steps, is_exo_3=False, epsilon=0.1) for _ in zero_to_nb_agents]
    agents_egreedy_e02 = [Agent('e-greedy', game, nb_steps, is_exo_3=False, epsilon=0.2) for _ in zero_to_nb_agents]

    agents_egreedy_t = [Agent('e-greedy', game, nb_steps, is_exo_3=True, epsilon=0.0) for _ in zero_to_nb_agents]
    agents_softmax_t = [Agent('softmax', game, nb_steps, is_exo_3=True, tau=1.0) for _ in zero_to_nb_agents]

    for k in zero_to_nb_agents:
        agents_random[k].learn()
        agents_softmax_1[k].learn()
        agents_softmax_01[k].learn()
        agents_egreedy_e00[k].learn()
        agents_egreedy_e01[k].learn()
        agents_egreedy_e02[k].learn()
        agents_egreedy_t[k].learn()
        agents_softmax_t[k].learn()
        print('agent', k, 'of each algorithm done')

    average_reward_for_each_algorithm(agents_egreedy_t,
                                      agents_softmax_t,
                                      agents_egreedy_e00,
                                      agents_egreedy_e01,
                                      agents_egreedy_e02,
                                      agents_random,
                                      agents_softmax_01,
                                      agents_softmax_1,
                                      nb_agents, nb_steps)

    generate_histograms(agents_egreedy_t,
                        agents_softmax_t,
                        agents_egreedy_e00,
                        agents_egreedy_e01,
                        agents_egreedy_e02,
                        agents_random,
                        agents_softmax_01,
                        agents_softmax_1,
                        nb_actions,
                        nb_agents)

    per_arm(agents_egreedy_t,
            agents_softmax_t,
            agents_egreedy_e00,
            agents_egreedy_e01,
            agents_egreedy_e02,
            agents_random,
            agents_softmax_01,
            agents_softmax_1,
            game,
            nb_actions,
            nb_agents,
            nb_steps,
            zero_to_nb_agents)


def per_arm(agents_egreedy_t, agents_softmax_t, agents_egreedy_e00, agents_egreedy_e01,
            agents_egreedy_e02, agents_random, agents_softmax_01,
            agents_softmax_1, game, nb_actions, nb_agents,
            nb_steps, zero_to_nb_agents):

    # average over time per arm
    arm_Q = [[], [], [], []]
    arms_random = [[], [], [], []]
    arms_softmax_1 = [[], [], [], []]
    arms_softmax_01 = [[], [], [], []]
    arms_egreedy_e00 = [[], [], [], []]
    arms_egreedy_e01 = [[], [], [], []]
    arms_egreedy_e02 = [[], [], [], []]

    arms_softmax_t = [[], [], [], []]
    arms_egreedy_t = [[], [], [], []]

    for arm in range(nb_actions):
        for s in range(nb_steps):
            avg_r = 0.0
            avg_soft_1 = 0.0
            avg_soft_01 = 0.0
            avg_egreedy_e00 = 0.0
            avg_egreedy_e01 = 0.0
            avg_egreedy_e02 = 0.0

            avg_egreedy_t = 0.0
            avg_softmax_t = 0.0

            for a in zero_to_nb_agents:
                avg_r += agents_random[a].get_stats()[3][arm][s]
                avg_soft_1 += agents_softmax_1[a].get_stats()[3][arm][s]
                avg_soft_01 += agents_softmax_01[a].get_stats()[3][arm][s]
                avg_egreedy_e00 += agents_egreedy_e00[a].get_stats()[3][arm][s]
                avg_egreedy_e01 += agents_egreedy_e01[a].get_stats()[3][arm][s]
                avg_egreedy_e02 += agents_egreedy_e02[a].get_stats()[3][arm][s]

                avg_egreedy_t += agents_egreedy_t[a].get_stats()[3][arm][s]
                avg_softmax_t += agents_softmax_t[a].get_stats()[3][arm][s]

            arms_random[arm].append(avg_r / nb_agents)
            arms_softmax_1[arm].append(avg_soft_1 / nb_agents)
            arms_softmax_01[arm].append(avg_soft_01 / nb_agents)
            arms_egreedy_e00[arm].append(avg_egreedy_e00 / nb_agents)
            arms_egreedy_e01[arm].append(avg_egreedy_e01 / nb_agents)
            arms_egreedy_e02[arm].append(avg_egreedy_e02 / nb_agents)

            arms_softmax_t[arm].append(avg_softmax_t / nb_agents)
            arms_egreedy_t[arm].append(avg_egreedy_t / nb_agents)

            arm_Q[arm].append(game.actions['q_ai*'][arm])

        x = range(nb_steps)
        list_of_legends = ['Actual estimation of Q_ai',
                           'Random',
                           'softmax with τ = 1',
                           'softmax with τ = 0.1',
                           'ε-Greedy with ε = 0',
                           'ε-Greedy with ε = 0.1',
                           'ε-Greedy with ε = 0.2',
                           'ε-Greedy with ε = 1/sqrt(t)',
                           'softmax with τ = 4*(1000-i)/1000']

        plots.multiplot([x, x, x, x, x, x, x, x, x], [arm_Q[arm],
                                                      arms_random[arm],
                                                      arms_softmax_1[arm],
                                                      arms_softmax_01[arm],
                                                      arms_egreedy_e00[arm],
                                                      arms_egreedy_e01[arm],
                                                      arms_egreedy_e02[arm],
                                                      arms_egreedy_t[arm],
                                                      arms_softmax_t[arm]],
                        'Action ' + str(arm) + ' - actual Q_ai = ' + str(game.actions['q_ai*'][arm]),
                        'Step',
                        'Q*',
                        list_of_legends)


def generate_histograms(agents_egreedy_t, agents_softmax_t, agents_egreedy_e00,
                        agents_egreedy_e01, agents_egreedy_e02,
                        agents_random, agents_softmax_01,
                        agents_softmax_1, nb_actions, nb_agents):
    average_bar_random = [0, 0, 0, 0]
    average_bar_softmax_1 = [0, 0, 0, 0]
    average_bar_softmax_01 = [0, 0, 0, 0]
    average_bar_egreedy_e00 = [0, 0, 0, 0]
    average_bar_egreedy_e01 = [0, 0, 0, 0]
    average_bar_egreedy_e02 = [0, 0, 0, 0]

    average_bar_egreedy_t = [0, 0, 0, 0]
    average_bar_softmax_t = [0, 0, 0, 0]

    for a in range(nb_agents):
        for k in range(nb_actions):
            average_bar_random[k] += agents_random[a].get_stats()[2][k]
            average_bar_softmax_1[k] += agents_softmax_1[a].get_stats()[2][k]
            average_bar_softmax_01[k] += agents_softmax_01[a].get_stats()[2][k]
            average_bar_egreedy_e00[k] += agents_egreedy_e00[a].get_stats()[2][k]
            average_bar_egreedy_e01[k] += agents_egreedy_e01[a].get_stats()[2][k]
            average_bar_egreedy_e02[k] += agents_egreedy_e02[a].get_stats()[2][k]
            average_bar_egreedy_t[k] += agents_egreedy_t[a].get_stats()[2][k]
            average_bar_softmax_t[k] += agents_softmax_t[a].get_stats()[2][k]

    for k in range(nb_actions):
        average_bar_random[k] /= nb_agents
        average_bar_softmax_1[k] /= nb_agents
        average_bar_softmax_01[k] /= nb_agents
        average_bar_egreedy_e00[k] /= nb_agents
        average_bar_egreedy_e01[k] /= nb_agents
        average_bar_egreedy_e02[k] /= nb_agents
        average_bar_egreedy_t[k] /= nb_agents
        average_bar_softmax_t[k] /= nb_agents

    plots.histogram(average_bar_random,
                    "Average number of time each action is selected (for 1000 agent) - Random action method",
                    "Actions",
                    "Amount of time the action is selected",
                    ["Action 1", "Action 2", "Action 3", "Action 4"])
    plots.histogram(average_bar_softmax_1,
                    "Average number of time each action is selected (for 1000 agent) - softmax τ = 1 action method",
                    "Actions",
                    "Amount of time the action is selected",
                    ["Action 1", "Action 2", "Action 3", "Action 4"])
    plots.histogram(average_bar_softmax_01,
                    "Average number of time each action is selected (for 1000 agent) - softmax τ = 0.1 action method",
                    "Actions",
                    "Amount of time the action is selected",
                    ["Action 1", "Action 2", "Action 3", "Action 4"])
    plots.histogram(average_bar_egreedy_e00,
                    "Average number of time each action is selected (for 1000 agent) - ε-Greedy ε = 0 action method",
                    "Actions",
                    "Amount of time the action is selected",
                    ["Action 1", "Action 2", "Action 3", "Action 4"])
    plots.histogram(average_bar_egreedy_e01,
                    "Average number of time each action is selected (for 1000 agent) - ε-Greedy ε = 0.1 action method",
                    "Actions",
                    "Amount of time the action is selected",
                    ["Action 1", "Action 2", "Action 3", "Action 4"])
    plots.histogram(average_bar_egreedy_e02,
                    "Average number of time each action is selected (for 1000 agent) - ε-Greedy ε = 0.2 action method",
                    "Actions",
                    "Amount of time the action is selected",
                    ["Action 1", "Action 2", "Action 3", "Action 4"])

    plots.histogram(average_bar_softmax_t,
                    "Average number of time each action is selected (for 1000 agent)"
                    " - softmax τ = 4*(1000-t)/1000 action method",
                    "Actions",
                    "Amount of time the action is selected",
                    ["Action 1", "Action 2", "Action 3", "Action 4"])

    plots.histogram(average_bar_egreedy_t,
                    "Average number of time each action is selected (for 1000 agent)"
                    " - ε-Greedy ε = 1/str(t) action method",
                    "Actions",
                    "Amount of time the action is selected",
                    ["Action 1", "Action 2", "Action 3", "Action 4"])


def average_reward_for_each_algorithm(agents_egreedy_t, agents_softmax_t,
                                      agents_egreedy_e00, agents_egreedy_e01, agents_egreedy_e02, agents_random,
                                      agents_softmax_01, agents_softmax_1, nb_agents, nb_steps):
    average_agents_random = []
    average_agents_softmax_1 = []
    average_agents_softmax_01 = []
    average_agents_egreedy_e00 = []
    average_agents_egreedy_e01 = []
    average_agents_egreedy_e02 = []

    average_agents_egreedy_t = []
    average_agents_softmax_t = []

    # stats
    for s in range(nb_steps):
        average_agent_random_step_i = 0.0
        average_agent_softmax_1_step_i = 0.0
        average_agent_softmax_01_step_i = 0.0
        average_agent_egreedy_e00_step_i = 0.0
        average_agent_egreedy_e01_step_i = 0.0
        average_agent_egreedy_e02_step_i = 0.0

        average_agent_egreedy_t_step_i = 0.0
        average_agent_softmax_t_step_i = 0.0

        for a in range(nb_agents):
            average_agent_random_step_i += agents_random[a].get_stats()[1][s]
            average_agent_softmax_1_step_i += agents_softmax_1[a].get_stats()[1][s]
            average_agent_softmax_01_step_i += agents_softmax_01[a].get_stats()[1][s]
            average_agent_egreedy_e00_step_i += agents_egreedy_e00[a].get_stats()[1][s]
            average_agent_egreedy_e01_step_i += agents_egreedy_e01[a].get_stats()[1][s]
            average_agent_egreedy_e02_step_i += agents_egreedy_e02[a].get_stats()[1][s]

            average_agent_egreedy_t_step_i += agents_egreedy_t[a].get_stats()[1][s]
            average_agent_softmax_t_step_i += agents_softmax_t[a].get_stats()[1][s]

        average_agents_random.append(average_agent_random_step_i / nb_agents)
        average_agents_softmax_1.append(average_agent_softmax_1_step_i / nb_agents)
        average_agents_softmax_01.append(average_agent_softmax_01_step_i / nb_agents)
        average_agents_egreedy_e00.append(average_agent_egreedy_e00_step_i / nb_agents)
        average_agents_egreedy_e01.append(average_agent_egreedy_e01_step_i / nb_agents)
        average_agents_egreedy_e02.append(average_agent_egreedy_e02_step_i / nb_agents)

        average_agents_egreedy_t.append(average_agent_egreedy_t_step_i / nb_agents)
        average_agents_softmax_t.append(average_agent_softmax_t_step_i / nb_agents)

    list_x = [range(nb_steps) for _ in range(8)]  # because 8 lines
    list_y = [average_agents_random,
              average_agents_softmax_1,
              average_agents_softmax_01,
              average_agents_egreedy_e00,
              average_agents_egreedy_e01,
              average_agents_egreedy_e02,

              average_agents_softmax_t,
              average_agents_egreedy_t]
    list_of_legends = ['Random',
                       'softmax with τ = 1',
                       'softmax with τ = 0.1',
                       'ε-Greedy with ε = 0',
                       'ε-Greedy with ε = 0.1',
                       'ε-Greedy with ε = 0.2',
                       'softmax with τ = 4*(1000-t)/1000',
                       'ε-Greedy with ε = 1/sqrt(t)']
    plots.multiplot(list_of_x=list_x,
                    list_of_y=list_y,
                    title='Average reward over ' + str(nb_steps) + ' steps ; ' + str(nb_agents) + ' agents',
                    xlabel='# runs',
                    ylabel='Average reward',
                    list_of_legends=list_of_legends)
