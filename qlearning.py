import numpy as np
import random 

# class Q_learning:
#     # Q-learning parameters

    
#     def __init__(self, gym, cart):
#         self.gym = gym
#         self.cart = cart

class Tabular_Q():
    def __init__(self, gym, cart, gamma, alpha, epsilon):
        # super().__init__(gym, cart)
    
        # The matrix
        Q = np.zeros([cart.STATE_SIZE, cart.ACTION_SIZE])

        print("Total matrix size: {}x{}".format(cart.STATE_SIZE, cart.ACTION_SIZE))
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
            state_index = cart.discretize(state)
            
            # We are done when the pole angle >= 12 degrees or we solved the problem
            while not done:
                if random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample() # Explore state space
                else:
                    action = np.argmax(Q[state_index]) # Exploit learned values

                next_state, reward, done, info = env.step(action) # invoke Gym
                next_state_index = cart.discretize(next_state)
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