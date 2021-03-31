import pandas as pd
import matplotlib.pyplot as plt
import datetime

class Experiment_episode_timesteps:
  def __init__(self, _columns):
    self.df = pd.DataFrame(columns = _columns)

  def Episode_time(self, episode, avg_timesteps):
    new_line = {}
    new_line["episodes"] = episode        
    new_line["avg_timesteps"] = avg_timesteps

    self.df = self.df.append(new_line, ignore_index=True)

  def Loss_reward(self, loss, reward, filename):
    
    df = pd.DataFrame({'losses':loss, 'reward':reward})

    print(df)

    # Add episodes
    df.insert(0, 'episodes', range(0, len(df)))
    
    # self.Create_line_plot(df, 'temp', 'temp')

    # Create some mock data
    # t = np.arange(0.01, 10.0, 0.01)
    data1 = loss
    data2 = reward

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Number of episodes')
    ax1.set_ylabel('Number of timesteps', color=color)
    ax1.plot(reward, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Loss', color=color)  # we already handled the x-label with ax1
    ax2.plot(loss, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('plots/{0}-{1}.png'.format(filename, datetime.datetime.now().strftime("%H:%M:%S")))

    plt.show()

    # exit(0)

  def Clear_df(self):
    self.df = self.df[0:0]

  def Plot_episode_timesteps(self, df):
    pass


  def Create_line_plot(self, df, filename, _title):
    """Simple function that creates a line plot of the given dataframe.

    Args:
        df (pd df): dataframe with TrueSkill scores of the bots
        filename (string): filename to be given
    """

    ax = df.plot.line(title=_title, x='episodes')
    
    ax.set_xlabel("Number of episodes")
    ax.set_ylabel("Number of timesteps")
    
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

    plt.xlim([0, df['episodes'].max()])
    # plt.ylim([0, trueskill_max])

    # To make X axis nice integers
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.savefig('plots/{0}-{1}.png'.format(filename, datetime.datetime.now().strftime("%H:%M:%S")))
    plt.show()