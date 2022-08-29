"""
Create wrappers for the inputs of the Atari Experiments

Modified from https://github.com/Neo-X/SMiRL_Code/blob/master/surprise/wrappers/obsresize.py#L305-L354 
"""


import gym
import numpy as np
from atari_kit.preprocessing import PreprocessAtari

# We want to create a set of environments for the agent to train. 
# In other words, the agent learns from multiple atari games
# in a single wrapper.

class SoftResetWrapper(gym.Env):
    def __init__(self, envs_list, max_time = 300, initial = 0):
        self._envs_list = envs_list
        self._time = 0
        self._max_time = max_time
        self._number_envs = envs_list.shape[0]
        self._env = envs_list[initial]
        
        # Gym spaces
        self.action_space = envs_list[initial].action_space
        self.observation_space = envs_list[initial].observation_space
    
    def step(self, action):
        # Take Action
        obs, env_rew, envdone, info = self._env.step(action)
        
        info["life_length_avg"] = self._last_death
        if (envdone):
            obs_ = self.reset()
            info["death"] = 1
            self._last_death = 0
            obs = np.random.rand(*obs_.shape)
        else:
            info["death"] = 0
        
        self._last_death = self._last_death + 1
        envdone = self._time >= self._max_time
        return obs, env_rew, envdone, info

    # Reset lets the agent to switch into a new Atari game.
    def reset(self):
        '''
        Reset the wrapped env and the buffer
        '''
        print(f'Before reset: {self._env}')
        self._time = 0
        self._last_death = 0
        obs = self._env.reset()

        for i in range(len(self._envs_list)):
            self._envs_list[i].reset()

        # We use this random generated number in order to 
        # pick the next Atari environment.
        random_number = np.random.randint(0, self._number_envs)
        print(f'Chosen random number: {random_number}')

        self._env = self._envs_list[random_number]
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        
        print(f'After reset: {self._env}')

        return obs
    
    def render(self, mode=None):
        return self._env.render(mode=mode)

# Create a set of environments while preprocessing them at the same time.
def create_set(name_envs):
    set_of_envs = np.empty([len(name_envs)], dtype=object)
    
    for i in range(len(name_envs)):
        env = gym.make(name_envs[i])
        env = PreprocessAtari(env)
        set_of_envs[i] = env

    return set_of_envs

# Create an environment (wrapper) for the Atari experiments.
# If we only have one game, that will be the environment.
# If we have multiple games, we will need to use the Atari Wrapper class.
def make_env(name_envs):
    if len(name_envs) == 1:
        env = gym.make(name_envs[0])
        env = PreprocessAtari(env)
    else:
        set_of_envs = create_set(name_envs)
        env = SoftResetWrapper(set_of_envs)
        
    return env
    
# Testing with the wrapper
if __name__ == "__main__":
    env = make_env(["BeamRiderNoFrameskip-v4", "RiverraidNoFrameskip-v4", "AssaultNoFrameskip-v4", "SpaceInvadersNoFrameskip-v4"])
    print(env.reset())
    print(env.step(0))
    print(env.reset())