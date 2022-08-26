"""
Run DQN on Atari environments
Choice of environments:
1. "Breakout-v0"
2. "Pong-v0"
3. "BeamRider-v0"
4. "Seaquest-v0"
"""

import gym
from torch import nn as nn
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.policies.argmax import ArgmaxDiscretePolicy
from rlkit.torch.dqn.dqn import DQNTrainer

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from atari_kit.preprocessing import PreprocessAtari
from atari_kit.wrappers import *
from rlkit.torch.networks.custom import ConvNet1, ConvNet2
from name_experiment import *
from doodad.easy_launch.python_function import run_experiment
from rlkit.core import logger
#import stable_baselines3.common.atari_wrappers as atari_wrappers
import numpy as np

#
#def make_env(env_name):
    #env = gym.make(env_name)
    #env = PreprocessAtari(env)
    # In Atari (preprocessed)
    # -> Image 84 x 84
    # -> 1 channel only
    # -> 4 actions

    # Atari Preprocessing Wrapper
    #env = gym.wrappers.AtariPreprocessing(env, noop_max = 30, frame_skip = 4,
    #                                 screen_size = 84, terminal_on_life_loss = False,
    #                                 grayscale_obs = True,
    #                                 grayscale_newaxis = False,
    #                                 scale_obs = False)

    # Frame stacking
    #env_list = np.empty([1], dtype=object)
    #env_list[0] = env

    #env = SoftResetWrapper(env_list)
    #env = gym.wrappers.FrameStack(env, 4)

    #return env
#"""

def experiment(doodad_config, variant):

    """
    print('doodad_config.base_log_dir: ', doodad_config.base_log_dir)
    setup_logger(f'DQN-{variant["atari_env"]}', variant=variant,
                 log_dir=doodad_config.base_log_dir)
    """
    print('doodad_config.base_log_dir: ', doodad_config.base_log_dir)
    name = variant["exp_name"]
    output_path = f'{doodad_config.base_log_dir}/{name}/{variant["mode"]}'

    setup_logger(name, variant=variant,
                 log_dir=output_path)
                # Example of log_dir for first experiment of Breakout-v0 with DQN Algorithm
                # log_dir: C:/Users/ronni/Documents/rlkit/data/DQN-Breakout/v0/exp1
    #setup_logger(f'DQN-{variant["atari_env"]}', variant=variant)

    #expl_env = make_env("Breakout-v0")
    #eval_env = make_env("Breakout-v0")

    #expl_env = make_env(["SpaceInvaders-v0"])
    #eval_env = make_env(["SpaceInvaders-v0"])

    expl_env = make_env(["SpaceInvaders-v0"])
    eval_env = make_env(["SpaceInvaders-v0"])

    print(f'ACTIONS FOR EXPL: {expl_env.action_space.n}')
    print(f'ACTIONS FOR EVAL: {eval_env.action_space.n}')

    #expl_env = make_env(["Breakout-v0", "BeamRider-v0"])
    #eval_env = make_env(["SpaceInvaders-v0"])

    expl_n_actions = expl_env.action_space.n

    qf = ConvNet1(expl_n_actions)
    target_qf = ConvNet1(expl_n_actions)

    qf_criterion = nn.MSELoss()
    eval_policy = ArgmaxDiscretePolicy(qf)
    expl_policy = PolicyWrappedWithExplorationStrategy(
        EpsilonGreedy(expl_env.action_space),
        eval_policy
    )
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        expl_policy,
    )
    trainer = DQNTrainer(
        qf=qf,
        target_qf=target_qf,
        qf_criterion=qf_criterion,
        **variant['trainer_kwargs']
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()

    print("Experiment complete!")
    print(f'Your experiment has been saved in the following directory:')
    print(output_path)

    print('-------------------------')
    print('GPU Information')
    import torch
    print(f'Is GPU detected? {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'How many GPUs? {torch.cuda.device_count()}')
        print(f'Device name? {torch.cuda.get_device_name(0)}')
        print(f'Compatibilities: {torch.cuda.get_arch_list()}')
        print(f'Version of Cuda: {torch.version.cuda}')

if __name__ == "__main__":
    #env_name = get_choice_env()

    # noinspection PyTypeChecker
    variant = dict(
        algorithm="DQN",
        version="normal",
        #mode="here_no_doodad",
        #mode="local",
        #mode="local_docker",
        mode="ssh",
        replay_buffer_size=int(1E5), #1E6 => 5e5
        algorithm_kwargs=dict(
            # Original num_epochs: 3000
            num_epochs=5000,
            # 5000 - 1000 - 1000 - 1000 - 1000 - 256
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=500, # now 500
            batch_size=32,
        ),
        trainer_kwargs=dict(
            discount=0.99, #0.99 initially
            learning_rate=0.00025 # 3E-4, #3e-4 initially
        ),
        exp_name='test5'
    )

    ptu.set_gpu_mode(True)
    run_experiment(experiment, 
        exp_name=variant["exp_name"], 
        use_gpu=True,
        #use_gpu=False,
        ssh_host='blue',
        variant=variant, mode=variant["mode"]
    )

