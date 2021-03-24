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
from mcts import Mcts
from experiments import Experiment_episode_timesteps




class Cart:
    # These are the limits we use for the observation values
    LIM_CART_POSITION = 2.4 # real: 2.4
    LIM_CART_VELOCITY = 1.0 # real: inf
    LIM_POLE_ANGLE = 12     # real: 48
    LIM_POLE_ROTATION = 25  # real: inf

    # These are the step sizes we use
    STEP_CART_POSITION = 0.01
    STEP_CART_VELOCITY = 0.01
    STEP_POLE_ANGLE = 1
    STEP_POLE_ROTATION = 1

    NUMSTEPS_CART_POSITION = int((LIM_CART_POSITION * 2) / STEP_CART_POSITION)
    NUMSTEPS_CART_VELOCITY = int((LIM_CART_VELOCITY * 2) / STEP_CART_VELOCITY)
    NUMSTEPS_POLE_ANGLE = int((LIM_POLE_ANGLE * 2) / STEP_POLE_ANGLE)
    NUMSTEPS_POLE_ROTATION = int((LIM_POLE_ROTATION * 2) / STEP_POLE_ROTATION)

    STATE_SIZE = NUMSTEPS_CART_POSITION * NUMSTEPS_CART_VELOCITY * NUMSTEPS_POLE_ANGLE * NUMSTEPS_POLE_ROTATION
    ACTION_SIZE = 2 # Two actions, 0=push_left, 1=push_right

    def discretize(self, observation) -> int: # Maybe reject values outside selected space, maybe increase precision
        """Map an observation to a unique index, to use for the Q-table

        Args:
            observation (ndarray[4]): [position of cart, velocity of cart, angle of pole, rotation rate of pole]

        Returns:
            int: index based on ordering of the elements in their respective spaces
        """    
        cart_position, cart_velocity, pole_angle, pole_rotation = observation
        
        # calculate orders first
        cart_position = int((round(cart_position, 2) + self.LIM_CART_POSITION) / self.STEP_CART_POSITION)
        cart_velocity = int((round(cart_velocity, 2) + self.LIM_CART_VELOCITY) / self.STEP_CART_VELOCITY)
        pole_angle = int((round(pole_angle, 0) + self.LIM_POLE_ANGLE) / self.STEP_POLE_ANGLE)
        pole_rotation = int((round(pole_rotation, 0) + self.LIM_POLE_ROTATION) / self.STEP_POLE_ROTATION)

        index = cart_position
        index = index + self.NUMSTEPS_CART_POSITION * cart_velocity
        index = index + self.NUMSTEPS_CART_POSITION * self.NUMSTEPS_CART_VELOCITY * pole_angle
        index = index + self.NUMSTEPS_CART_POSITION * self.NUMSTEPS_CART_VELOCITY * self.NUMSTEPS_POLE_ANGLE * pole_rotation

        return index


def main(argv):

    # Function parameters
    gamma = 0.7     # discount factor
    alpha = 0.2     # learning rate
    epsilon = 0.1   # epsilon greedy
    
    iterations = 1e6

    # Substantiate Cart in order to pass to other functions
    cart = Cart()
    # Substantiate an experiment
    exp = Experiment_episode_timesteps(["episodes", "avg_timesteps"])
    # Substantiate the different programs
    random_actions = Random_actions()
    tabular_q = Tabular_Q()
    deep_q = Deep_Q()
    mcts = Mcts()

    for i, arg in enumerate(argv):

        if arg == "random":
            result = random_actions.main(gym, exp, iterations)

        elif arg == "tabular":
            result = tabular_q.main(gym, exp, cart, gamma, alpha, epsilon, iterations)

        elif arg == "deep":
            result = deep_q.main(gym, exp, cart, gamma, alpha, epsilon, iterations)

        elif arg == "mcts":
            result = mcts.main(gym, exp, cart)

        elif arg == "exp1":
            # Here we pitch Random versus Tabular
            exp = Experiment_episode_timesteps(["episodes", "avg_timesteps"])
            df_rnd = random_actions.main(gym, exp, iterations)

            exp = Experiment_episode_timesteps(["episodes", "avg_timesteps"])
            df_tab = tabular_q.main(gym, exp, cart, gamma, alpha, epsilon, iterations)

            # Fix the dataframe names
            df_rnd = df_rnd.rename({'avg_timesteps': 'random'}, axis=1)
            df_tab = df_tab.rename({'avg_timesteps': 'tabular'}, axis=1)

            result = pd.merge(df_rnd, df_tab, how="inner", on='episodes')

            exp.Create_line_plot(result, 'filename', 'Random versus Tabular')

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
            # Pitch Tabular versus Deep
            pass

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