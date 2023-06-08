import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces.box import Box

import os

from gymnasium.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
#import gymnasium.ActionWrapper as ActionWrapper

import stable_baselines3 as baselines
import stable_baselines3.common.monitor as monitor
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, EpisodicLifeEnv, FireResetEnv, \
       MaxAndSkipEnv, NoopResetEnv, StickyActionEnv, WarpFrame # import all wrappers

#from collections import deque
#import cv2
#cv2.ocl.setUseOpenCL(False)

# Not used. Use make_atari_env instead.
def make_atari(env_id, max_episode_steps=None):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30) # this make the environment stochastic by sampling an initial state.
           # However there are two parameters in s-b3 of make `frameskip`, `repeat_action_probability` which does the same.
           # Actually, this also seems to be unnecessary, since in gymnasium there is a seed parameter in reset.
    env = MaxAndSkipEnv(env, skip=4) # skip every n-th frame and takes max of two last. Very similar to `frameskip` parameter 
    # in s-b3 of make. So probably not needed.
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps) # this is not needed in s-b3 because it is an argument to make
    return env

# Not used. Use make_atari_env instead.
def wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=False):
    """Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings(): # this seems to be useful to reduce the action space.
        env = FireResetEnv(env)
        
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
        
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env

def make_atari_env(env_id, frameskip=4, repeat_action_probability=0.25, max_episode_steps=None, 
                   init_noop_max=None, episode_life=True, remove_fire_action=True, new_size=None, 
                   make_greyscale=False, clip_rewards=True, num_frame_stack=None):
    
    # Defauls for `frameskip`, `repeat_action_probability` for pong-v5 are: 4, 0.25
    env = gym.make(env_id, render_mode='rgb_array', frameskip=frameskip, repeat_action_probability=repeat_action_probability,
                   max_episode_steps=max_episode_steps)
    
    if init_noop_max is not None:
        env = NoopResetEnv(env, noop_max=init_noop_max) # this make the environment stochastic by sampling an initial state.
           # However there are two parameters in s-b3 of make `frameskip`, `repeat_action_probability` which does the same.
           # Actually, this also seems to be unnecessary, since in gymnasium there is a seed parameter in reset.
    
    if episode_life:
        # seems to be the same to `terminal_on_life_loss` parameter of AtariPreprocessing in Gymnasium
        env = EpisodicLifeEnv(env)
        
    if remove_fire_action:
        # ok this does not work as I thouhgt it should.
        # See example of action space reductin in https://github.com/openai/retro/blob/master/retro/examples/discretizer.py#L9
        #if 'FIRE' in env.unwrapped.get_action_meanings(): # this seems to be useful to reduce the action space.
        #    env = FireResetEnv(env)
        tt1 = [ (not ('FIRE' in mm), ii) for (ii, mm) in enumerate(env.unwrapped.get_action_meanings()) ]
        tt2 = [ii for (bb, ii) in tt1 if (bb is True) ]
        new_to_old_map = { new_ind:old_ind for new_ind, old_ind in enumerate(tt2) }
        
        env = ReduceActions(env, new_to_old_map)
        
        
    if (new_size is not None):
        #env = WarpFrame(env, width=new_size, height=new_size, grayscale=True)
        env = ResizeObservation(env, (new_size, new_size))
        
    if make_greyscale:
        env = GrayScaleObservation(env, keep_dim = True)
        
    if clip_rewards:
        env = ClipRewardEnv(env)
        
    if (num_frame_stack is not None):
        env = FrameStack(env, num_frame_stack) # default is 4
    
    return env

def obs_fit_shape_to_pytorch(obs, extra_batch_dim=False):
    """
    obs (np.ndarray or LazyFrame): observation of shape (stacked_frames * width * height * color_dim) 
    """
    
    #import pdb; pdb.set_trace()
    obs = np.array(obs, copy=False) # convert to array if input is of type LazyFrame
    orig_shape = obs.shape
    
    # To feed into Pytorch convolutions we need share (num_channels * width * height)
    if len(orig_shape) == 3: # No stacked observations
        new_obs = np.transpose(obs, axes = (2, 0, 1))
    elif len(orig_shape) == 4: # Stacked observations
        tt1 = np.transpose(obs, axes = (0, 3, 1, 2))
        new_obs = np.reshape(obs, (-1, tt1.shape[-2], tt1.shape[-1]))
        #new_obs = tt1
    else:
        import pdb; pdb.set_trace()
    
    if extra_batch_dim:
        new_obs = new_obs[np.newaxis,:]
            
    return new_obs

class WrapShapePyTorch(gym.ObservationWrapper):
    """
    Wrapper for applying function `obs_fit_shape_to_pytorch`. 
    Converts observation shape to the one required by Pytorch conv_2d.
    """
    
    def __init__(self, env=None, extra_batch_dim=False):
        super(WrapShapePyTorch, self).__init__(env)
        
        self.extra_batch_dim = extra_batch_dim
        self.orig_obs_shape_len = len(self.observation_space.shape)
        orig_obs_shape = self.observation_space.shape
        
        if len(orig_obs_shape) == 3: # no FrameStack
            new_shape = (orig_obs_shape[2], orig_obs_shape[0], orig_obs_shape[1])
            low = self.observation_space.low[0, 0, 0]
            high = self.observation_space.high[0, 0, 0]
        elif len(orig_obs_shape) == 4: # with FrameStack
            new_shape = (orig_obs_shape[0]*orig_obs_shape[3], orig_obs_shape[1], orig_obs_shape[2])
            low = self.observation_space.low[0, 0, 0, 0]
            high = self.observation_space.high[0, 0, 0, 0]
        else:
            raise ValueError('Unsupported observation shape')
        
        if self.extra_batch_dim:
            new_shape = (1,) + new_shape
        
        
        self.observation_space = Box(
            low,
            high,
            new_shape,
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return obs_fit_shape_to_pytorch(observation, extra_batch_dim=self.extra_batch_dim)

    
class ImageToPyTorch(gym.ObservationWrapper):
    """
    Image shape to num_channels x weight x height
    """
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.uint8)

    def observation(self, observation):
        return np.swapaxes(observation, 2, 0)/255.0
    
def wrap_pytorch(env):
    return ImageToPyTorch(env)

class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)

def make_env_a2c_atari(env_id, seed, rank, log_dir):
    def _thunk():
        env = make_atari(env_id)
        env.seed(seed + rank)

        obs_shape = env.observation_space.shape

        if log_dir is not None:
            env = monitor.Monitor(env, os.path.join(log_dir, str(rank)))

        env = wrap_deepmind(env)

        obs_shape = env.observation_space.shape
        env = WrapPyTorch(env)

        return env
    return _thunk

class ReduceActions(gym.ActionWrapper):
    def __init__(self, env, new_to_old_map):
        super().__init__(env)
        self.new_to_old_map = new_to_old_map
        self.action_space = spaces.Discrete(len(new_to_old_map))

    def action(self, act):
        return self.new_to_old_map[act]



# Copied from baselines
class LazyFrames_baselines(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]

# Copied from baselines.
# Defines new wrapper
class FrameStack_baselines(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        # ob, reward, done, info = self.env.step(action) AlexGrig
        ob, reward, terminated, truncated, info = self.env.step(action)
        
        self.frames.append(ob)
        return self._get_ob(), reward, terminated, truncated, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


# Copied from baselines.
# Defines new wrapper    
class WarpFrame_baselines(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.

        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs