# Inspired from https://github.com/raillab/dqn

# path to orig module when `run_train` is located separately
import sys
module_path = '/Users/alexgrig/Yandex.Disk.localized/Programming/python/dqn-atari-gymnasium'
if module_path not in sys.path:
    sys.path.append(module_path)
    
import time
import random
import numpy as np
import gymnasium as gym

from dql.agent import DQNAgent
from dql.replay_buffer import ReplayBuffer
import utils.plotting as plotting

#import dql.wrappers_old_gym.py as wrap

from pathlib import Path

import utils.wrappers_gymnasium as wrap
import utils.log as log
import torch
import argparse

import config.config as config

def make_env(env_id='PongNoFrameskip-v4', remove_fire_action=False, record_video=False, log_dir = None):
    #env_id = "PongNoFrameskip-v4"
    env = wrap.make_atari_env(env_id, frameskip=4, repeat_action_probability=0, max_episode_steps=None, 
                   init_noop_max=30, episode_life=True, remove_fire_action=remove_fire_action, new_size=84, 
                   make_greyscale=True, clip_rewards=True, num_frame_stack=4)
    env = wrap.WrapShapePyTorch(env, extra_batch_dim=False)
    
    if record_video:
        env.metadata["render_fps"] = 10
        env.metadata["fps"] = 10
        env = wrap.RecordVideo(env, video_folder=log_dir, episode_trigger=lambda episode_no: True, 
                   step_trigger = None, video_length = 0, name_prefix = 'test-video', disable_logger = False)
        # Note! to be able to use video recorder like this I had to modify external code in two places:
        # 1) File: python3.9/site-packages/moviepy/video/VideoClip.py
        #    I commented one decorator of the method `write_videofile` because it could not get default 
        #       and explicitely transferred fps param.:
        #    @requires_duration
        #    #@use_clip_fps_by_default
        #    @convert_masks_to_RGB
        #    def write_videofile(self, filename, fps=None, codec=None, ...
        # 2) File site-packages/gymnasium/wrappers/monitoring/video_recorder.py  
        #    In method `close` I changed the call to `write_videofile` function to be able to explicitely
        #    transfer `fps` parameter.
        #    # AlexGrig ->
        #    # clip.write_videofile(self.path, logger=moviepy_logger
        #    clip.write_videofile(self.path, fps=self.frames_per_sec, logger=moviepy_logger)
        #    # AlexGrig <-
        # Only after this manipulations vedeo recording started to work.     
    print(env_id)
    return env

def make_agent(env, replay_buffer, checkpoint_file=None, use_double_dqn=True, lr=0.001, batch_size=32, gamma=0.99, 
               device='cpu', dqn_type='nature'):
    
    agent = DQNAgent(
        env.observation_space,
        env.action_space,
        replay_buffer,
        use_double_dqn=use_double_dqn,
        lr=lr,
        batch_size=batch_size,
        gamma=gamma,
        device=device,
        dqn_type=dqn_type
    )

    if(checkpoint_file):
        print(f"Loading a policy - { checkpoint_file } ")
        agent.policy_network.load_state_dict(torch.load(checkpoint_file))
    
    return agent

def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    
def compute_epsilon(step_no, fraction_of_total, total_steps_num, epsilon_start, epsilon_end):
    """
    Function computes a linearly decreasing learning rate from `epsilon_start` to `epsilon_end`.
    The decrease is done for a `fraction_of_total` fraction of total timesteps. Then is remains at
    `epsilon_end`.
    
    step_no (int): current step
    fraction_of_total (float): fraction of total number of steps in which to reach the minimum epsilon
    total_steps_num (int): total number of steps
    epsilon_start (float): initial epsilon
    epsilon_end (float): final epsilon
    """
    
    timesteps_to_min = fraction_of_total * float(total_steps_num)
    
    fraction = min(1.0, float(step_no) / timesteps_to_min)
    epsilon_threshold = epsilon_start + fraction * (epsilon_end - epsilon_start)
    
    return epsilon_threshold

def get_state(obs):
    """
    Function which alters the observation received from the environment. Output from this function is written
    to replay_buffer.
    """
    
    #state = np.array(obs)
    #state = state.transpose((2, 0, 1))
    #state = torch.from_numpy(state)
    #return state.unsqueeze(0)
    if isinstance(obs, tuple): # seems to work on reset in gymansium
        tt0 = obs[0]
    else:
        tt0 = obs
    #tt1 = wrap.obs_fit_shape_to_pytorch(tt0, extra_batch_dim=False)
    #return torch.from_numpy(tt1)
    return tt0

def train(env, agent, log_dir = './run_log', total_steps_num=10000, learning_starts=1000, learning_freq=1, 
          target_update_freq=100, epsilon_start=1, epsilon_end=0.01, epsilon_fraction=0.1, 
          action_log_frequency=None, print_freq=None, fig_plot_freq=None):    
    """
    Traininf of RL agent
    
    total_steps_num (int): total number of steps
    learning_starts (int): step at which learning begins
    learning_freq (int): frequency of policy network update
    target_update_freq (int): frequency of target network update
    
    """
    
    # define loggers:
    action_log = (Path(log_dir) / 'action_log.csv').absolute()
    action_selections = [0 for _ in range(env.action_space.n)]
    action_logger = log.ActionLogger(action_selections, action_log, action_log_frequency)
    
    step_log = (Path(log_dir) / 'step_log.csv').absolute()
    step_logger = log.Logger(step_log, erase_existing=True)
    
    episode_log = (Path(log_dir) / 'episode_log.csv').absolute()
    episode_logger = log.Logger(episode_log, erase_existing=True)
    
    # Env reset:
    state = get_state( env.reset() )
    
    episode_rewards = [0.0]
    episode_lengths = [0.0]
    episode_no = 0
    td_loss = None
    
    train_start_time = time.time()
    episode_start_time = train_start_time
    step_start_time = train_start_time
    for step in range(1, total_steps_num+1):
        eplison_threshold = compute_epsilon(step, epsilon_fraction, total_steps_num, 
                                            epsilon_start, epsilon_end)
        
        sample = random.random()
        if(sample > eplison_threshold):
            # Exploit
            action = agent.act(state)
        else:
            # Explore
            action = env.action_space.sample()
        
        #next_state, reward, done, info = env.step(action)
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = get_state(next_state)
        done = (terminated or truncated)
        
        agent.memory.add(state, action, reward, next_state, float(done)) 
        # later float(done) is used in loss computation. (These records are avoided)
        
        state = next_state

        episode_rewards[-1] += reward
        episode_lengths[-1] += 1
        
        # Logging ->
        # We want to log here terminal states as well.
        step_logger.log({'step_no': step, 'td_loss': td_loss, 'reward': reward, 'action': action, 'done': done})
        if (action_log_frequency is not None): # frequency is checked in the logger
            action_logger.log_action(step, action)
        # Logging <-
        
        if done: # episode ends
            episode_no += 1
            # Logging ->
            episode_logger.log({'episode_no': episode_no,'step_no': step, 'episode_reward': episode_rewards[-1], 'episode_length': int(episode_lengths[-1]), 'episode_time': (time.time() - episode_start_time),  'time_since_train_start_min': (time.time() - train_start_time)/60 })
            # Logging <-
            state = get_state( env.reset() )
            
            episode_start_time = time.time()
            episode_rewards.append(0.0)
            episode_lengths.append(0.0)
            #import pdb; pdb.set_trace()
            
        if step > learning_starts and step % learning_freq == 0:
            td_loss = agent.optimise_td_loss()
        else:
            td_loss = None
        
        if step > learning_starts and (step % target_update_freq == 0):
            agent.update_target_network()        
        
        
        # Logging ->
        if (fig_plot_freq is not None) and (step % fig_plot_freq == 0):
            plotting.plot_all_data(step_logger.log_file, episode_logger.log_file, action_logger.log_file, 
                                   step, total_steps_num, ipynb = False, log_plot_file_name = 'log.jpg')
            
        if done and (print_freq is not None) and ((episode_no % print_freq) == 0):
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            print("********************************************************")
            print("steps: {}".format(step))
            print("episodes: {}".format(episode_no))
            print("last episode reward: {}".format(episode_rewards[-1]))
            print("mean 100 episode reward: {}".format(mean_100ep_reward))
            print("Epsilon: {}".format(eplison_threshold))
            print("% time spent exploring: {}".format(int(100 * eplison_threshold)))
            print("********************************************************")
            torch.save(agent.policy_network.state_dict(), Path(log_dir) / f'checkpoint.pth')
         # Logging <-  
        
    # TODO: rename the checkpoint to include episode_num and step num.
    # TODO: include interafce for both toolboxes gym and dymnasium e.g. in set_state function.

def test(env, agent, log_dir):
    
    # define loggers:
    step_log = (Path(log_dir) / 'test_step_log.csv').absolute()
    step_logger = log.Logger(step_log, erase_existing=True)
    
    state = get_state( env.reset() )
    
    episode_reward = 0
    episode_length = 0
    
    episode_start_time = time.time()
    step = 0
    done = False
    while not done:
        action = agent.act(state)
        
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = get_state(next_state)
        done = (terminated or truncated)
        
        state = next_state
        
        step += 1
        episode_reward += reward
        episode_length += 1
        
        # We want to log terniminal states as well.
        step_logger.log({'step_no': step, 'reward': reward, 'action': action, 'done': done})
    
        if done: # episode ends
            env.close()
            print(f'Test episode finished.')
            print(f'Episode reward: {episode_reward}')
            print(f'Episode length: {episode_length}')
            print(f'Episode time: {time.time() - episode_start_time}')
            break
            
if __name__ == '__main__':

    # Process command line params:   
    parser = argparse.ArgumentParser(description='DQN Atari')
    
    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('--load-checkpoint-file', type=str, default=None, 
                        help='Where checkpoint file should be loaded from (usually results/checkpoint.pth)')
    parser.add_argument('--test', action='count', default=0, 
                        help='Whether test of 1 episode should be run. Assumes `--load-checkpoint-file` to be present.')
    parser.add_argument('--num-threads', type=int, default=None, 
                        help='Number of threads fed into `torch.set_num_threads()` ')
    args = parser.parse_args()
    
    hyper_params = config.read_yaml(args.config)
    print(hyper_params)
    # Set num threads:
    
    test_mode=False
    if args.test > 0:
        test_mode = True
        if not (args.load_checkpoint_file):
            raise ValueError('You have requested testing but have not provided a checkpoint file.')
        
    if (not test_mode) and (args.num_threads is not None):
        print(f'num_threads: {args.num_threads}')
        torch.set_num_threads(args.num_threads)
    
    # If you have a checkpoint file, spend less time exploring:
    if (not test_mode) and (args.load_checkpoint_file):
        hyper_params['learning_params']['epsilon_start']= 0.01
    else:
        pass
    print(f"Start epsilon: { hyper_params['learning_params']['epsilon_start'] }")
    
    # Old code of gym PongNoFrameskip-v4" for info ->
    #hyper_params = {
    #    "seed": 42,  # which seed to use
    #    "env": "PongNoFrameskip-v4",  # name of the game
    #    "replay-buffer-size": int(5e3),  # replay buffer size
    #    "learning-rate": 1e-4,  # learning rate for Adam optimizer
    #    "discount-factor": 0.99,  # discount factor
    #    "dqn_type": "nature", # neurips
    #    # total number of steps to run the environment for
    #    "num-steps": int(1e6),
    #    "batch-size": 32,  # number of transitions to optimize at the same time
    #    "learning-starts": 10000,  # number of steps before learning starts
    #    "learning-freq": 1,  # number of iterations between every optimization step
    #    "use-double-dqn": True,  # use double deep Q-learning
    #    "target-update-freq": 1000,  # number of iterations between every target network update
    #    "eps-start": eps_start,  # e-greedy start threshold
    #    "eps-end": 0.01,  # e-greedy end threshold
    #    "eps-fraction": 0.1,  # fraction of num-steps
    #    "print-freq": 10
    #}

    #assert "NoFrameskip" in hyper_params["env"], "Require environment with no frameskip"
    #env = gym.make(hyper_params["env"])
    #env.seed(hyper_params["seed"])

    #env = NoopResetEnv(env, noop_max=30)
    #env = MaxAndSkipEnv(env, skip=4)
    #env = EpisodicLifeEnv(env)
    #env = FireResetEnv(env)
    #env = WarpFrame(env)
    #env = PyTorchFrame(env)
    #env = ClipRewardEnv(env)
    #env = FrameStack(env, 4)
    #env = gym.wrappers.Monitor(
    #    env, './video/', video_callable=lambda episode_id: episode_id % 50 == 0, force=True)
    # Old code of gym PongNoFrameskip-v4" for info <-
    
    #make_env(env_id='PongNoFrameskip-v4', remove_fire_action=False, record_video=False, video_dir = None)
    set_seeds(**hyper_params["randomness_params"])
    if test_mode:
        log_dir = Path(args.load_checkpoint_file).absolute().parent
        env = make_env(**hyper_params["env_params"], record_video=True, log_dir=log_dir)
        
    else:
        env = make_env(**hyper_params["env_params"])
    
    #import pdb; pdb.set_trace()
    
    params_for_replay_buffer = hyper_params["agent_params"].filter_elements_with_func_named_args(ReplayBuffer.__init__)
    replay_buffer = ReplayBuffer(**params_for_replay_buffer)
    
    params_for_dqn_agent = hyper_params["agent_params"].filter_elements_with_func_named_args(make_agent)
    agent = make_agent(
        env,
        replay_buffer,
        checkpoint_file = Path(args.load_checkpoint_file).absolute(), 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        **params_for_dqn_agent)
    
    if test_mode:
        #params_for_testing = hyper_params["log_params"].filter_elements_with_func_named_args(test)
        test(env, agent, log_dir=log_dir)
    else:
        params_for_training = hyper_params["learning_params"].filter_elements_with_func_named_args(train) + hyper_params["log_params"]
        train(env, agent, **params_for_training)

## Running pdb postmortem:
    
#import sys
#import pdb
#
#def pdb_postmortem(exc_type, exc_value, exc_traceback):
#    pdb.post_mortem(exc_traceback)
#
## Set the excepthook to the custom pdb_postmortem function
#sys.excepthook = pdb_postmortem
#
## Your code here...