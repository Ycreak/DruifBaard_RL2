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

import gym
# import numpy as np
# import random

from random_action import Random_actions
from qlearning import *

# pip3 install -r requirements.txt

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


def main():

    d = Cart()
    # Call the random_actions simulation
    # Random_actions(gym)

    gamma = 0.7     # discount factor
    alpha = 0.2     # learning rate
    epsilon = 0.1   # epsilon greedy

    Tabular_Q(gym, d, gamma, alpha, epsilon)
    # Q_learning()
    exit()

if __name__ == "__main__":
    main()