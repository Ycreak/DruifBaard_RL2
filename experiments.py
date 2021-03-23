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

  def Clear_df(self):
    self.df = self.df[0:0]

  def Plot_episode_timesteps(self, df):
    pass

  def Create_line_plot(self, df, filename):
    """Simple function that creates a line plot of the given dataframe.

    Args:
        df (pd df): dataframe with TrueSkill scores of the bots
        filename (string): filename to be given
    """
    from matplotlib.ticker import MaxNLocator

    # Y Cap.
    trueskill_max = 50

    # print(df)

    df = df.drop(['episodes'], axis=1) #TODO: this should be the index!

    ax = df.plot.line(title='Line plot')
    
    ax.set_xlabel("Number of episodes (in thousands)")
    ax.set_ylabel("Number of timesteps")
    
    # plt.xlim([0, self.tourney_rounds])
    # plt.ylim([0, trueskill_max])

    # To make X axis nice integers
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # plt.show()
    plt.savefig('plots/{0}-{1}.png'.format(filename, datetime.datetime.now().strftime("%H:%M:%S")))