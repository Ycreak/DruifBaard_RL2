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
import numpy as np
import random


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

# Q-learning parameters
gamma = 0.7     # discount factor
alpha = 0.2     # learning rate
epsilon = 0.1   # epsilon greedy


def discretize(observation) -> int: # Maybe reject values outside selected space, maybe increase precision
    """Map an observation to a unique index, to use for the Q-table

    Args:
        observation (ndarray[4]): [position of cart, velocity of cart, angle of pole, rotation rate of pole]

    Returns:
        int: index based on ordering of the elements in their respective spaces
    """    
    cart_position, cart_velocity, pole_angle, pole_rotation = observation
    
    # calculate orders first
    cart_position = int((round(cart_position, 2) + LIM_CART_POSITION) / STEP_CART_POSITION)
    cart_velocity = int((round(cart_velocity, 2) + LIM_CART_VELOCITY) / STEP_CART_VELOCITY)
    pole_angle = int((round(pole_angle, 0) + LIM_POLE_ANGLE) / STEP_POLE_ANGLE)
    pole_rotation = int((round(pole_rotation, 0) + LIM_POLE_ROTATION) / STEP_POLE_ROTATION)

    index = cart_position
    index = index + NUMSTEPS_CART_POSITION * cart_velocity
    index = index + NUMSTEPS_CART_POSITION * NUMSTEPS_CART_VELOCITY * pole_angle
    index = index + NUMSTEPS_CART_POSITION * NUMSTEPS_CART_VELOCITY * NUMSTEPS_POLE_ANGLE * pole_rotation
    return index

def Q_learning():
    # The matrix
    Q = np.zeros([STATE_SIZE, ACTION_SIZE])

    print("Total matrix size: {}x{}".format(STATE_SIZE, ACTION_SIZE))
    print("Estimated GB used: {}GB".format(Q.nbytes/1000000000))
    print_per_n = 1000

    env = gym.make('CartPole-v0')
    env.reset()

    # observation --> 4-tuple: [position of cart, velocity of cart, angle of pole, rotation rate of pole]
    # reward --> always 1: for every step the pole does not fall
    # done --> boolean: becomes true if the pole is >12degrees
    # info --> for debugging

    total_timesteps = 0
    total_timesteps_n_episodes = 0
    for episode in range(1000000):
        done = False
        t = 0

        # Initialize state
        state = env.reset()
        state_index = discretize(state)
        
        # We are done when the pole angle >= 12 degrees or we solved the problem
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore state space
            else:
                action = np.argmax(Q[state_index]) # Exploit learned values

            next_state, reward, done, info = env.step(action) # invoke Gym
            next_state_index = discretize(next_state)
            next_max = np.max(Q[next_state_index])
            old_value = Q[state_index, action]
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            Q[state_index, action] = new_value
            state = next_state
            state_index = next_state_index
            t = t + 1
            total_timesteps = total_timesteps + 1
            total_timesteps_n_episodes = total_timesteps_n_episodes + 1

        # Print statistics
        if episode % print_per_n == 0:
            print("Episode {} finished after {} timesteps".format(episode, t+1))
            print("Average timesteps {}".format(total_timesteps / (episode+1)))
            print("Average timesteps of last {} episodes: {}\n".format(print_per_n, total_timesteps_n_episodes / print_per_n))
            total_timesteps_n_episodes = 0
    env.close()

# For comparison
def random_actions():
    env = gym.make('CartPole-v0')
    env.reset()

    total_timesteps = 0
    for episode in range(1000000):
        done = False
        t = 0
        
        # Initialize state
        state = env.reset()

        while not done:
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            state = next_state
            t = t + 1
            total_timesteps = total_timesteps + 1
        # Print statistics
        if episode % 1000 == 0:
            print("Episode {} finished after {} timesteps".format(episode, t+1))
            print("Average timesteps {}\n".format(total_timesteps / (episode+1)))
        
    env.close()

def main():
    #random_actions()
    Q_learning()
    exit()

if __name__ == "__main__":
    main()