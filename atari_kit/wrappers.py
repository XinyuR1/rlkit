import gym
import numpy as np
from atari_kit.preprocessing import PreprocessAtari

# Taken from https://github.com/Neo-X/SMiRL_Code/blob/master/surprise/wrappers/obsresize.py#L305-L354 

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
            ### Trick to make "death" more surprising...
            #   info["life_length"] = self._last_death
            info["death"] = 1
            self._last_death = 0
            obs = np.random.rand(*obs_.shape)
        else:
            info["death"] = 0
        
        self._last_death = self._last_death + 1
        envdone = self._time >= self._max_time
        return obs, env_rew, envdone, info

    def reset(self):
        '''
        Reset the wrapped env and the buffer
        '''
        print(f'Before reset: {self._env}')
        self._time = 0
        self._last_death = 0
        obs = self._env.reset()

        random_number = np.random.randint(0, self._number_envs)
        print(f'Chosen random number: {random_number}')

        self._env = self._envs_list[random_number]
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        
        print(f'After reset: {self._env}')

        return obs
    
    def render(self, mode=None):
        return self._env.render(mode=mode)

        # Keep the step function and for reset, you pick another environment using the random number

def create_set(name_envs):
    set_of_envs = np.empty([len(name_envs)], dtype=object)
    
    for i in range(len(name_envs)):
        env = gym.make(name_envs[i])
        env = PreprocessAtari(env)
        set_of_envs[i] = env

    return set_of_envs

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
    env = make_env(["Assault-v0", "BeamRider-v0", "Riverraid-v0"])
    print(env.reset())
    env = make_env(["SpaceInvaders-v0"])
    print(env.reset())