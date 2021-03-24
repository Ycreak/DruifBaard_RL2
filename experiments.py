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