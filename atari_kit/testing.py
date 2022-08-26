import gym
import stable_baselines3.common.atari_wrappers as atari_wrappers

name = "BreakoutNoFrameskip-v4"
env = gym.make(name)

# Atari Preprocessing Wrapper
env = gym.wrappers.AtariPreprocessing(env, noop_max = 30, frame_skip = 4,
                                     screen_size = 84, terminal_on_life_loss = False,
                                     grayscale_obs = True,
                                     grayscale_newaxis = False,
                                     scale_obs = False)

# Frame stacking
env = gym.wrappers.FrameStack(env, 4)
env = atari_wrappers.ClipRewardEnv(env)

print(f'\nName of the environment: {name}')
print(f'Actions: {env.unwrapped.get_action_meanings()}\n')