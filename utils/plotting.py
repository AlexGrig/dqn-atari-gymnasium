from pathlib import Path
import csv

import numpy as np
import pandas as pd

#switch backend in driver file
import matplotlib
import matplotlib.pyplot as plt


def load_step_data(step_log_file, step_no_field='step_no', loss_field='td_loss', reward_field='reward'): # load reward and loss data
    
    step_log_file = Path(step_log_file)
    df = pd.read_csv(step_log_file)
    
    xx = df[step_no_field].tolist()
    yy_loss = df[loss_field].tolist()
    
    return (xx, yy_loss), df

def plot_step_data(ax3, step_log_file, total_steps_no):

    step_log_file = Path(step_log_file)
    
    try:
        (xx, yy_loss), df = load_step_data(step_log_file, step_no_field='step_no', loss_field='td_loss', reward_field='reward')
    except OSError as err:
        return
    
    p3, = ax3.plot(xx, yy_loss, label="TD_LOSS")
    
    ax3.set_ylabel('TD Loss')
    ax3.yaxis.label.set_color(p3.get_color())
    ax3.tick_params(axis='y', colors=p3.get_color())

    ax3.legend([p3], [p3.get_label()], loc=4)
    ax3.set_title('Step TD Loss vs timestep')
    
    
def load_episode_data(episode_log_file, step_no_field='step_no', reward_field='episode_reward',
        episode_len_field='episode_length', episode_time_sec_field='episode_time'): # load reward and loss data
    
    episode_log_file = Path(episode_log_file)
    df = pd.read_csv(episode_log_file)
    
    
    xx = df[step_no_field].tolist()
    yy_reward = df[reward_field].tolist()
    yy_eplen = df[episode_len_field].tolist()
    
    
    return (xx, yy_reward), (xx, yy_eplen), df


def plot_episode_data(ax1, ax2, episode_log_file, total_steps_no):
    
    episode_log_file = Path(episode_log_file)
    
    try:
        (xx, yy_reward), (xx, yy_eplen), df = load_episode_data(episode_log_file, step_no_field='step_no',
                        reward_field='episode_reward',episode_len_field='episode_length', 
                        episode_time_sec_field='episode_time')
    except OSError as err:
        return

    if xx is None or yy_reward is None:
        return
    
    # plot
    ax1.set_ylabel('Episode Reward')
    p1, = ax1.plot(xx, yy_reward, label="Reward")

    ax1.yaxis.label.set_color(p1.get_color())
    ax1.tick_params(axis='y', colors=p1.get_color())

    ax1.legend([p1], [p1.get_label()], loc=4)
    ax1.set_title('Episode Reward vs timestep')
    
    
    ax2.set_ylabel('Length (steps)')
    p2, = ax2.plot(xx, yy_eplen, label="Episode length")
    
    ax2.yaxis.label.set_color(p2.get_color())
    ax2.tick_params(axis='y', colors=p2.get_color())

    ax2.legend([p2], [p2.get_label()], loc=4)
    ax2.set_title('Episode length vs timestep')
    
def load_action_data(action_log_file):
    
    action_log_file = Path(action_log_file)
    df = pd.read_csv(action_log_file, header=None)
    
    xx = df.iloc[:, 0].tolist()
    list_actions = df.iloc[:,1:].T.to_numpy().tolist() # action_num, len
    return xx, list_actions

def plot_action_data(ax4, action_log_file, total_steps_no):
    
    action_log_file = Path(action_log_file)
    try:
        xx, yys = load_action_data(action_log_file)
    except OSError as err:
        return
    
    ax4.set_ylabel('Action Selection Frequency(%)')
    labels = ['Action {}'.format(i) for i in range(len(yys))]
    p4 = ax4.stackplot(xx, yys, labels=labels)

    base = 0.0
    for percent, index in zip(yys, range(len(yys))):
        offset = base + percent[-1]/3.0
        ax4.annotate(str('{:.2f}'.format(yys[index][-1])), xy=(xx[-1], offset), color=p4[index].get_facecolor().ravel())
        base += percent[-1]

    ax4.legend(loc=4) #remake g2 legend because we have a new line
    ax4.set_title('Action Selection Frequency(%) vs Timestep')    
    
    
def plot_all_data(step_log, episode_log, action_log, step_no, total_steps_no, ipynb = False, log_plot_file_name = 'log.jpg'):
    episode_log = Path(episode_log)
    step_log = Path(step_log)
    action_log = Path(action_log)
    
    title = '' #None
    
    save_filename = Path(episode_log.parent / log_plot_file_name)
    
    matplotlib.rcParams.update({'font.size': 20})
    params = {
        'xtick.labelsize': 20,
        'ytick.labelsize': 15,
        'legend.fontsize': 15
    }
    plt.rcParams.update(params)
    
    tick_fractions = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    ticks = tick_fractions * total_steps_no
    tick_names = ["{:.0e}".format(tick) for tick in ticks]
    
    #if time is not None:
    #    title = 'Avg. Last 10 Rewards: ' +  str(np.round(np.mean(ty[-10]))) + ' || ' +  game + ' || Elapsed Time: ' + str(time)
    #else:
    #    title = 'Avg. Last 10 Rewards: ' +  str(np.round(np.mean(ty[-10]))) + ' || ' +  game
    title = f'Current timestep no: {step_no}'
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20, 30), subplot_kw = dict(xticks=ticks, xlim=(0, total_steps_no*1.15), xlabel='Timestep', title=title))
   
    fig.suptitle(title, fontsize=28)
    
    ax1.set_xticklabels(tick_names)
    ax2.set_xticklabels(tick_names)
    ax3.set_xticklabels(tick_names)
    ax4.set_xticklabels(tick_names)
    
    plot_episode_data(ax1, ax2, episode_log, total_steps_no)
    
    plot_step_data(ax3, step_log, total_steps_no)
    
    plot_action_data(ax4, action_log, total_steps_no)
    
    plt.tight_layout() # prevent label cutoff
    
    if ipynb:
        plt.show()
    else:
        plt.savefig(save_filename)
    plt.clf()
    plt.close()
    
    
if __name__ == '__main__':
    
    episode_log = Path('../log/episode_log.csv').absolute()
    step_log = Path('../log/step_log.csv').absolute()
    action_log = Path('../log/action_log.csv').absolute()
    
    total_steps_no = 1e6
    title = '' #None
    ipynb = False
    
    plot_all_data(step_log, episode_log, action_log, 150000, total_steps_no, ipynb = ipynb, log_plot_file_name = 'log.jpg')