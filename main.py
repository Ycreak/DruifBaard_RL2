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
import time
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

    # Initialises np with zeroes if true, otherwise with both zeroes and ones
    zeroes = True
    # Number of iterations for all implementations
    iterations = 1e6
    # Maximum amount of steps per episode (used in MCPG)
    max_steps = 1e4 
    #TODO: ???
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

        elif arg == "exp_rnd_tab":
            # Here we pitch Random versus Tabular
            exp = Experiment_episode_timesteps(["episodes", "avg_timesteps"])
            df_rnd = random_actions.main(gym, exp, iterations)
            df_rnd = df_rnd.drop(['avg_timesteps_last'], axis=1) 

            exp = Experiment_episode_timesteps(["episodes", "avg_timesteps"])
            df_tab = tabular_q.main(gym, exp, cart, gamma, alpha, epsilon, zeroes, iterations)
            df_tab = df_tab.drop(['avg_timesteps_last'], axis=1) 

            # Fix the dataframe names
            df_rnd = df_rnd.rename({'avg_timesteps': 'random'}, axis=1)
            df_tab = df_tab.rename({'avg_timesteps': 'tabular'}, axis=1)

            result = pd.merge(df_rnd, df_tab, how="inner", on='episodes')

            exp.Create_line_plot(result, 'Tabular_vs_Random', 'Random versus Tabular')

        elif arg == "exp_zeroes":
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

        elif arg == "exp_large":
            # Here we compare a small table with a larger table
            exp = Experiment_episode_timesteps(["episodes", "avg_timesteps"])
            df_tab = tabular_q.main(gym, exp, cart, gamma, alpha, epsilon, True, 1e6)
            df_tab = df_tab.rename({'avg_timesteps': 'tab1'}, axis=1)
            df_tab = df_tab.drop(['avg_timesteps_last'], axis=1) 
            # Change the parameters and run again
            cart = Cart(_STEP_POLE_ANGLE = 0.1, _STEP_POLE_ROTATION = 0.1, _round_parameter = 1)
   
            exp = Experiment_episode_timesteps(["episodes", "avg_timesteps"])
            df_tab2 = tabular_q.main(gym, exp, cart, gamma, alpha, epsilon, True, 1e6)
            df_tab2 = df_tab2.rename({'avg_timesteps': 'tab2'}, axis=1)
            df_tab2 = df_tab2.drop(['avg_timesteps_last'], axis=1) 
            # Merge results
            result = pd.merge(df_tab, df_tab2, how="inner", on='episodes')

            # Plot the line
            exp.Create_line_plot(result, 'Tabular_versus_Larger', 'Larger Table runs')

        elif arg == "exp_tab_e":
            # Here we compare results if parameters are tweaked
            result = exp.df # To allow merging of only one dataframe

            epsilon_list = [0.01, 0.1, 1]
            
            for epsilon in epsilon_list:
                # Start new experiment and run it
                exp = Experiment_episode_timesteps(["episodes", "avg_timesteps"])
                df_tab = tabular_q.main(gym, exp, cart, gamma, alpha, epsilon, True, 1e5)
                df_tab = df_tab.drop(['avg_timesteps_last'], axis=1) 

                # Provide the correct collumn name
                ep_col_name = 'e_' + str(epsilon)
                df_tab = df_tab.rename({'avg_timesteps': ep_col_name}, axis=1)
                # Merge the new df with the old one
                result = pd.merge(df_tab, result, how="left", on='episodes')
            # Drop the column we do not need    
            try:
                result = result.drop(['avg_timesteps'], axis=1) 
            except:
                print('no drop avg_timesteps')
            # Create the plot
            exp.Create_line_plot(result, 'epsilon_experiment', 'Tweaking Epsilon for Tabular')
            result = result[0:0]

        elif arg == "exp_tab_a":
            result = exp.df
            alpha_list = [0.2, 0.5, 0.8, 1]

            for alpha in alpha_list:
                # Start new experiment and run it
                exp = Experiment_episode_timesteps(["episodes", "avg_timesteps"])
                df_tab = tabular_q.main(gym, exp, cart, gamma, alpha, epsilon, True, 1e5)
                df_tab = df_tab.drop(['avg_timesteps_last'], axis=1) 

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
            exp.Create_line_plot(result, 'alpha_experiment', 'Tweaking Alpha for Tabular')
            result = result[0:0]

        elif arg == "exp_tab_g":
            result = exp.df
            gamma_list = [0.1, 0.5, 0.7, 0.9]

            for gamma in gamma_list:
                # Start new experiment and run it
                exp = Experiment_episode_timesteps(["episodes", "avg_timesteps"])
                df_tab = tabular_q.main(gym, exp, cart, gamma, alpha, epsilon, True, 1e5)
                df_tab = df_tab.drop(['avg_timesteps_last'], axis=1) 

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
            exp.Create_line_plot(result, 'gamma_experiment', 'Tweaking Gamma for Tabular')
            result = result[0:0]

        elif arg == "exp_deep":
            # Here we run an experiment with Deep-Q learning.            

            start_time = time.time()
            # result, losses, reward = deep_q.main(gym, exp, cart, gamma, alpha=1e-3, epsilon=epsilon, iterations=iterations)
            result, losses, reward = deep_q.main(gym, exp, cart, gamma, alpha=1e-3, epsilon=epsilon, iterations=1e4)
            deep_time = time.time() - start_time

            # Create an episode/timesteps plot
            exp.Create_line_plot(result, 'deepq', 'Deep-Q Learning')
            # Create an loss/reward plot
            exp.Loss_reward(losses, reward, 'deepq_loss')

            print(deep_time)

        elif arg == "exp_deep_e":
            # Here we compare results if parameters are tweaked
            result = exp.df # To allow merging of only one dataframe

            epsilon_list = [0.01, 0.1, 1]
            
            for epsilon in epsilon_list:
                # Start new experiment and run it
                exp = Experiment_episode_timesteps(["episodes", "avg_timesteps"])
                df_tab, losses, reward = deep_q.main(gym, exp, cart, gamma, alpha=1e-3, epsilon=epsilon, iterations=1e4)
                df_tab = df_tab.drop(['avg_timesteps_last'], axis=1) 
                
                # Provide the correct collumn name
                ep_col_name = 'e_' + str(epsilon)
                df_tab = df_tab.rename({'avg_timesteps': ep_col_name}, axis=1)
                # Merge the new df with the old one
                result = pd.merge(df_tab, result, how="left", on='episodes')
            # Drop the column we do not need    
            try:
                result = result.drop(['avg_timesteps'], axis=1) 
            except:
                print('no drop avg_timesteps')
            # Create the plot
            exp.Create_line_plot(result, 'epsilon_experiment_deep', 'Tweaking Epsilon for Deep')
            result = result[0:0]

        elif arg == "exp_deep_a":
            # Here we compare results if parameters are tweaked
            result = exp.df # To allow merging of only one dataframe

            alpha_list = [1e-1, 1e-2, 1e-3, 1e-4]
            
            for alpha in alpha_list:
                # Start new experiment and run it
                exp = Experiment_episode_timesteps(["episodes", "avg_timesteps"])
                df_tab, losses, reward = deep_q.main(gym, exp, cart, gamma, alpha=alpha, epsilon=epsilon, iterations=1e4)
                df_tab = df_tab.drop(['avg_timesteps_last'], axis=1) 
                
                # Provide the correct collumn name
                ep_col_name = 'a_' + str(alpha)
                df_tab = df_tab.rename({'avg_timesteps': ep_col_name}, axis=1)
                # Merge the new df with the old one
                result = pd.merge(df_tab, result, how="left", on='episodes')
            # Drop the column we do not need    
            try:
                result = result.drop(['avg_timesteps'], axis=1) 
            except:
                print('no drop avg_timesteps')
            # Create the plot
            exp.Create_line_plot(result, 'alpha_experiment_deep', 'Tweaking Alpha for Deep')
            result = result[0:0]

        elif arg == "exp_deep_g":
            # Here we compare results if parameters are tweaked
            result = exp.df # To allow merging of only one dataframe

            gamma_list = [0.1, 0.5, 0.7, 0.9]
            
            for gamma in gamma_list:
                # Start new experiment and run it
                exp = Experiment_episode_timesteps(["episodes", "avg_timesteps"])
                df_tab, losses, reward = deep_q.main(gym, exp, cart, gamma, alpha=1e-3, epsilon=epsilon, iterations=1e4)
                df_tab = df_tab.drop(['avg_timesteps_last'], axis=1) 
                
                # Provide the correct collumn name
                ep_col_name = 'g_' + str(gamma)
                df_tab = df_tab.rename({'avg_timesteps': ep_col_name}, axis=1)
                # Merge the new df with the old one
                result = pd.merge(df_tab, result, how="left", on='episodes')
            # Drop the column we do not need    
            try:
                result = result.drop(['avg_timesteps'], axis=1) 
            except:
                print('no drop avg_timesteps')
            # Create the plot
            exp.Create_line_plot(result, 'gamma_experiment_deep', 'Tweaking Gamma for Deep')
            result = result[0:0]            

        elif arg == "exp_mcpg":
            result = mcpg.main(gym, exp, cart, alpha=3e-4, gamma=0.9, iterations=5000, max_steps=10000, hidden_size=hidden_size)
            print(result)
            # Create an episode/timesteps plot
            exp.Create_line_plot(result, 'mcpg', 'Monte Carlo Policy Gradient')

        elif arg == "exp_mcpg_a":
            # Here we compare results if parameters are tweaked
            result = exp.df # To allow merging of only one dataframe

            alpha_list = [3e-2, 3e-3, 3e-4, 3e-5]
            
            for alpha in alpha_list:
                # Start new experiment and run it
                exp = Experiment_episode_timesteps(["episodes", "avg_timesteps"])
                df_tab = mcpg.main(gym, exp, cart, alpha=alpha, gamma=0.9, iterations=5000, max_steps=10000, hidden_size=hidden_size)
                df_tab = df_tab.drop(['avg_timesteps_last'], axis=1) 
                
                # Provide the correct collumn name
                ep_col_name = 'a_' + str(alpha)
                df_tab = df_tab.rename({'avg_timesteps': ep_col_name}, axis=1)
                # Merge the new df with the old one
                result = pd.merge(df_tab, result, how="left", on='episodes')
            # Drop the column we do not need    
            try:
                result = result.drop(['avg_timesteps'], axis=1) 
            except:
                print('no drop avg_timesteps')
            # Create the plot
            exp.Create_line_plot(result, 'alpha_experiment_mcpg', 'Tweaking Alpha for MCPG')
            result = result[0:0]

        elif arg == "exp_mcpg_g":
            # Here we compare results if parameters are tweaked
            result = exp.df # To allow merging of only one dataframe

            gamma_list = [0.1, 0.5, 0.7, 0.9]
            
            for gamma in gamma_list:
                # Start new experiment and run it
                exp = Experiment_episode_timesteps(["episodes", "avg_timesteps"])
                df_tab = mcpg.main(gym, exp, cart, alpha=3e-4, gamma=gamma, iterations=5000, max_steps=10000, hidden_size=hidden_size)
                df_tab = df_tab.drop(['avg_timesteps_last'], axis=1) 
                
                # Provide the correct collumn name
                ep_col_name = 'g_' + str(gamma)
                df_tab = df_tab.rename({'avg_timesteps': ep_col_name}, axis=1)
                # Merge the new df with the old one
                result = pd.merge(df_tab, result, how="left", on='episodes')
            # Drop the column we do not need    
            try:
                result = result.drop(['avg_timesteps'], axis=1) 
            except:
                print('no drop avg_timesteps')
            # Create the plot
            exp.Create_line_plot(result, 'gamma_experiment_MCPG', 'Tweaking Gamma for MCPG')
            result = result[0:0]   

        elif arg == "exp_mcpg_hidden":
            # Here we compare results if parameters are tweaked
            result = exp.df # To allow merging of only one dataframe

            hidden_list = [64, 128, 256, 512]
            
            for hidden in hidden_list:
                # Start new experiment and run it
                exp = Experiment_episode_timesteps(["episodes", "avg_timesteps"])
                df_tab = mcpg.main(gym, exp, cart, alpha=3e-4, gamma=0.9, iterations=5000, max_steps=10000, hidden_size=hidden)
                df_tab = df_tab.drop(['avg_timesteps_last'], axis=1) 
                
                # Provide the correct collumn name
                ep_col_name = 'hidden_' + str(hidden)
                df_tab = df_tab.rename({'avg_timesteps': ep_col_name}, axis=1)
                # Merge the new df with the old one
                result = pd.merge(df_tab, result, how="left", on='episodes')
            # Drop the column we do not need    
            try:
                result = result.drop(['avg_timesteps'], axis=1) 
            except:
                print('no drop avg_timesteps')
            # Create the plot
            exp.Create_line_plot(result, 'hidden_experiment_mcpg', 'Tweaking Hidden Size for MCPG')
            result = result[0:0]

        elif arg == "exp_all":
            # Run all implementations against eachother
            exp = Experiment_episode_timesteps(["episodes", "avg_timesteps"])
            start_time = time.time()
            df_mcpg = mcpg.main(gym, exp, cart, alpha=3e-4, gamma=0.9, iterations=5000, max_steps=10000, hidden_size=256)
            mcpg_time = time.time() - start_time

            exp = Experiment_episode_timesteps(["episodes", "avg_timesteps"])
            start_time = time.time()
            df_deep, losses, reward = deep_q.main(gym, exp, cart, gamma, alpha=1e-3, epsilon=epsilon, iterations=5000, _print_per_n=100)
            deep_time = time.time() - start_time

            exp = Experiment_episode_timesteps(["episodes", "avg_timesteps"])
            start_time = time.time()
            df_tab = tabular_q.main(gym, exp, cart, 0.9, 0.8, 0.1, True, 5000, _print_per_n=100)
            tabular_time = time.time() - start_time

            df_mcpg = df_mcpg.drop(['avg_timesteps_last'], axis=1) 
            df_deep = df_deep.drop(['avg_timesteps_last'], axis=1) 
            df_tab = df_tab.drop(['avg_timesteps_last'], axis=1) 

            df_mcpg = df_mcpg.rename({'avg_timesteps': 'mcpg'}, axis=1)
            df_tab = df_tab.rename({'avg_timesteps': 'tabular'}, axis=1)
            df_deep = df_deep.rename({'avg_timesteps': 'deep'}, axis=1)
            

            # Merge the new df with the old one
            result = pd.merge(df_mcpg, df_deep, how="left", on='episodes')
            result = pd.merge(result, df_tab, how="left", on='episodes')

            print(result)
            print(mcpg_time, deep_time, tabular_time)
            
            # Create the plot
            exp.Create_line_plot(result, 'deep_mcpg', 'Tabular versus DeepQ versus MCPG')

        else:
            print("Invalid argument.")
            exit(1)


if __name__ == "__main__":
    print('Hello')
    main(sys.argv[1:])
