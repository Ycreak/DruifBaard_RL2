class Random_actions:

    def __init__(self, gym):

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