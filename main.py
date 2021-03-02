# On Mac:
#   pip3 install gym
#   brew install glfw3
#   brew install glew

# observation --> 4-tuple: [position of cart, velocity of cart, angle of pole, rotation rate of pole]
# reward --> always 1: for every step the pole does not fall
# done --> boolean: becomes true if the pole is >12degrees
# info --> for debugging

import gym
import numpy as np
import random


LIM_CART_POSITION = 2.4
LIM_CART_VELOCITY = 1.0
LIM_POLE_ANGLE = 12.0
LIM_POLE_ROTATION = 25

STEP_CART_POSITION = 0.1
STEP_CART_VELOCITY = 0.01
STEP_POLE_ANGLE = 0.1
STEP_POLE_ROTATION = 0.01

NUMSTEPS_CART_POSITION = int((LIM_CART_POSITION * 2) / STEP_CART_POSITION)
NUMSTEPS_CART_VELOCITY = int((LIM_CART_VELOCITY * 2) / STEP_CART_VELOCITY)
NUMSTEPS_POLE_ANGLE = int((LIM_POLE_ANGLE * 2) / STEP_POLE_ANGLE)
NUMSTEPS_POLE_ROTATION = int((LIM_POLE_ROTATION * 2) / STEP_POLE_ROTATION)

STATE_SIZE = NUMSTEPS_CART_POSITION * NUMSTEPS_CART_VELOCITY * NUMSTEPS_POLE_ANGLE * NUMSTEPS_POLE_ROTATION
ACTION_SIZE = 2

gamma = 0.7 # discount factor
alpha = 0.2 # learning rate
epsilon = 0.1 # epsilon greedy


def discretize(observation) -> int: # Maybe reject values outside selected space, maybe increase precision
    cart_position, cart_velocity, pole_angle, pole_rotation = observation
    
    # calculate orders first
    cart_position = int((round(cart_position, 1) + LIM_CART_POSITION) / STEP_CART_POSITION)
    cart_velocity = int((round(cart_velocity, 2) + LIM_CART_VELOCITY) / STEP_CART_VELOCITY)
    pole_angle = int((round(pole_angle, 1) + LIM_POLE_ANGLE) / STEP_POLE_ANGLE)
    pole_rotation = int((round(pole_rotation, 1) + LIM_POLE_ROTATION) / STEP_POLE_ROTATION)

    index = cart_position
    index = index + NUMSTEPS_CART_POSITION * cart_velocity
    index = index + NUMSTEPS_CART_VELOCITY * pole_angle
    index = index + NUMSTEPS_POLE_ANGLE * pole_rotation
    return index

# The matrix
Q = np.zeros((STATE_SIZE, ACTION_SIZE))

env = gym.make('CartPole-v0')
env.reset()

total_timesteps = 0
for episode in range(100000):
    done = False
    total_reward = 0
    state = env.reset()
    t = 0
    while not done:
        #env.render()
        state_index = discretize(state)
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
        total_reward += reward
        state = next_state
        t = t + 1
        total_timesteps = total_timesteps + 1
    if episode % 100 == 0:
        print("Episode finished after {} timesteps".format(t+1))
        print("Average timesteps {}".format(total_timesteps / (episode+1)))
env.close()