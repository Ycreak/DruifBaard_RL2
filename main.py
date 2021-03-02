# On Mac:
#   pip3 install gym
#   brew install glfw3
#   brew install glew

# observation --> 4-tuple: [position of cart, velocity of cart, angle of pole, rotation rate of pole]
# reward --> always 1: for every step the pole does not fall
# done --> boolean: becomes true if the pole is certainly going to fall
# info --> for debugging

import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()