# Reinforcement Learning
# Leiden University 2021
# Assignment 2
# Philippe Bors S1773585
# Job van der Zwaag S1893378
# Luuk Nolden S1370898

# Cartpole api: https://github.com/openai/gym/wiki/CartPole-v0

# What to install on Mac:
#   pip3 install gym
#   brew install glfw3
#   brew install glew

# To install on Linux, run the following:
#   $ pip3 install -r requirements.txt

# Library Imports
import gym
import sys , getopt
import pandas as pd

# Class Imports
from random_action import Random_actions
from qlearning import Tabular_Q, Deep_Q
from mcpg import Mcpg, Policy_Network
from utils import Cart
from experiments import Experiment_episode_timesteps

def main(argv):

    # Function parameters
    gamma = 0.7     # discount factor
    alpha = 0.2     # learning rate 
    epsilon = 0.1   # epsilon greedy
    
    zeroes = False

    iterations = 1e6
    max_steps = 1e4 
    hidden_size = 128

    # Substantiate Cart in order to pass to other functions
    cart = Cart()
    # Substantiate an experiment
    exp = Experiment_episode_timesteps(["episodes", "avg_timesteps"])
    # Substantiate the different programs
    random_actions = Random_actions()
    tabular_q = Tabular_Q()
    deep_q = Deep_Q()
    mcpg = Mcpg()

    for i, arg in enumerate(argv):

        if arg == "random":
            result = random_actions.main(gym, exp, iterations)

        elif arg == "tabular":
            result = tabular_q.main(gym, exp, cart, gamma, alpha, epsilon, zeroes, iterations)
            exp.Create_line_plot(result, 'filename', 'Tabular')

        elif arg == "deep":
            result, losses, reward = deep_q.main(gym, exp, cart, gamma, alpha=1e-3, epsilon=epsilon, iterations=iterations)

        elif arg == "mcpg":
            result = mcpg.main(gym, exp, cart, alpha=3e-4, gamma=0.9, iterations=5000, max_steps=10000, hidden_size=hidden_size)

        elif arg == "exp1":
            # Here we pitch Random versus Tabular
            exp = Experiment_episode_timesteps(["episodes", "avg_timesteps"])
            df_rnd = random_actions.main(gym, exp, iterations)

            exp = Experiment_episode_timesteps(["episodes", "avg_timesteps"])
            df_tab = tabular_q.main(gym, exp, cart, gamma, alpha, epsilon, zeroes, iterations)

            # Fix the dataframe names
            df_rnd = df_rnd.rename({'avg_timesteps': 'random'}, axis=1)
            df_tab = df_tab.rename({'avg_timesteps': 'tabular'}, axis=1)

            result = pd.merge(df_rnd, df_tab, how="inner", on='episodes')

            exp.Create_line_plot(result, 'Tabular_vs_Random', 'Random versus Tabular')

        elif arg == "exp1_zeroes":
            # Here we pitch Tabular with zeroes vs Tabular with zeroes and ones
            exp = Experiment_episode_timesteps(["episodes", "avg_timesteps"])
            df_ones = tabular_q.main(gym, exp, cart, gamma, alpha, epsilon, False, iterations)

            exp = Experiment_episode_timesteps(["episodes", "avg_timesteps"])
            df_zeroes = tabular_q.main(gym, exp, cart, gamma, alpha, epsilon, True, iterations)

            # Fix the dataframe names
            df_ones = df_ones.rename({'avg_timesteps': 'ones'}, axis=1)
            df_zeroes = df_zeroes.rename({'avg_timesteps': 'zeroes'}, axis=1)

            result = pd.merge(df_ones, df_zeroes, how="inner", on='episodes')

            exp.Create_line_plot(result, 'Zeroes_vs_Ones', 'Zeroes versus Zeroes and Ones')

        elif arg == "exp1_large":
            # Run the experiment with a small table
            exp = Experiment_episode_timesteps(["episodes", "avg_timesteps"])
            df_tab = tabular_q.main(gym, exp, cart, gamma, alpha, epsilon, zeroes, iterations)
            df_tab = df_tab.rename({'avg_timesteps': 'tab1'}, axis=1)
            # Change the parameters and run again

            cart = Cart(_STEP_POLE_ANGLE = 0.1, _STEP_POLE_ROTATION = 0.1, _round_parameter = 1)
   
            exp = Experiment_episode_timesteps(["episodes", "avg_timesteps"])
            df_tab2 = tabular_q.main(gym, exp, cart, gamma, alpha, epsilon, zeroes, iterations)
            df_tab2 = df_tab2.rename({'avg_timesteps': 'tab2'}, axis=1)
            # Merge results
            result = pd.merge(df_tab, df_tab2, how="inner", on='episodes')

            # Plot the line
            exp.Create_line_plot(result, 'Tabular_versus_Larger', 'Larger Table runs')

        elif arg == "exp2_e":
            # Here we compare results if parameters are tweaked
            result = exp.df # To allow merging of only one dataframe

            epsilon_list = [0.01, 0.1, 1]
            
            for epsilon in epsilon_list:
                # Start new experiment and run it
                exp = Experiment_episode_timesteps(["episodes", "avg_timesteps"])
                df_tab = tabular_q.main(gym, exp, cart, gamma, alpha, epsilon, iterations)
                # Provide the correct collumn name
                ep_col_name = 'e_' + str(epsilon)
                df_tab = df_tab.rename({'avg_timesteps': ep_col_name}, axis=1)
                # Merge the new df with the old one
                result = pd.merge(df_tab, result, how="left", on='episodes')
            # Drop the column we do not need    
            result = result.drop(['avg_timesteps'], axis=1) 
            # Create the plot
            exp.Create_line_plot(result, 'filename', 'Tweaking Epsilon')
            result = result[0:0]

        elif arg == "exp2_a":
            result = exp.df
            alpha_list = [0.2, 0.5, 0.8]

            for alpha in alpha_list:
                # Start new experiment and run it
                exp = Experiment_episode_timesteps(["episodes", "avg_timesteps"])
                df_tab = tabular_q.main(gym, exp, cart, gamma, alpha, epsilon, iterations)
                # Provide the correct collumn name
                a_col_name = 'a_' + str(alpha)
                df_tab = df_tab.rename({'avg_timesteps': a_col_name}, axis=1)
                # Merge the new df with the old one
                result = pd.merge(df_tab, result, how="left", on='episodes')
            # Drop the column we do not need    
            try:
                result = result.drop(['avg_timesteps'], axis=1) 
            except:
                print('no drop avg_timesteps')
            # Create the plot
            exp.Create_line_plot(result, 'alpha_experiment', 'Tweaking Alpha')
            result = result[0:0]

        elif arg == "exp2_g":
            result = exp.df            
            gamma_list = [0.1, 0.5, 0.7]

            for gamma in gamma_list:
                # Start new experiment and run it
                exp = Experiment_episode_timesteps(["episodes", "avg_timesteps"])
                df_tab = tabular_q.main(gym, exp, cart, gamma, alpha, epsilon, iterations)
                # Provide the correct collumn name
                g_col_name = 'g_' + str(gamma)
                df_tab = df_tab.rename({'avg_timesteps': g_col_name}, axis=1)
                # Merge the new df with the old one
                result = pd.merge(df_tab, result, how="left", on='episodes')
            # Drop the column we do not need    
            try:
                result = result.drop(['avg_timesteps'], axis=1) 
            except:
                print('no drop avg_timesteps')            
            # Create the plot
            exp.Create_line_plot(result, 'gamma_experiment', 'Tweaking Gamma')
            result = result[0:0]

            # exit(0)
        elif arg == "exp3":
            # Deep.            
            result, losses, reward = deep_q.main(gym, exp, cart, gamma, alpha=1e-3, epsilon=epsilon, iterations=iterations)
            
            # Create an episode/timesteps plot
            exp.Create_line_plot(result, 'deepq', 'Deep-Q Learning')
            # Create an loss/reward plot
            exp.Loss_reward(losses, reward, 'deepq_loss')
            
            # pass

        elif arg == "exp4":
            # MCTS and tweaks
            pass

        elif arg == "exp5":
            # MCTS versus Q
            pass

        elif arg == "exp6":
            # Placeholder
            pass

        else:
            print("Invalid argument.")
            exit(1)

        print(result)

if __name__ == "__main__":
      
    main(sys.argv[1:])