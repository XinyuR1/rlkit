"""
Run DQN on CartPole-v0.
"""

from atari_kit.preprocessing import PreprocessAtari
import gym
from rlkit.core.simple_offline_rl_algorithm import SimpleOfflineRlAlgorithm
from rlkit.torch.networks.custom import ConvNet2
from torch import nn as nn

from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.policies.argmax import ArgmaxDiscretePolicy
from rlkit.torch.dqn.dqn import DQNTrainer
from rlkit.torch.networks import Mlp
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from doodad.easy_launch.python_function import run_experiment

def make_env(env_name):
    env = gym.make(env_name)
    env = PreprocessAtari(env)
    
    return env


def experiment(doodad_config, variant):
    print('doodad_config.base_log_dir: ', doodad_config.base_log_dir)
    name = variant["exp_name"]
    output_path = f'{doodad_config.base_log_dir}/{name}/{variant["mode"]}'

    setup_logger(name, variant=variant,
                 log_dir=output_path)

    expl_env = gym.make('CartPole-v0').env
    eval_env = gym.make('CartPole-v0').env

    #expl_env = make_env("SpaceInvaders-v0")
    #eval_env = make_env("SpaceInvaders-v0")

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.n

    qf = Mlp(
        hidden_sizes=[32, 32],
        input_size=obs_dim,
        output_size=action_dim,
    )
    target_qf = Mlp(
        hidden_sizes=[32, 32],
        input_size=obs_dim,
        output_size=action_dim,
    )
    
    qf_criterion = nn.MSELoss()
    eval_policy = ArgmaxDiscretePolicy(qf)
    expl_policy = PolicyWrappedWithExplorationStrategy(
        EpsilonGreedy(expl_env.action_space),
        eval_policy,
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


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        exp_name="Cartpole-v0",
        algorithm="DQN",
        version="normal",
        mode="ssh",
        layer_size=256,
        replay_buffer_size=int(1E5),# originally 1e6
        algorithm_kwargs=dict(
            num_epochs=5000, #originally 3000
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=500,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            learning_rate=3E-4,
        ),
    )

    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    #experiment(variant)

    run_experiment(experiment,
                   exp_name=variant["exp_name"],
                   variant=variant,
                   mode=variant["mode"],
                   use_gpu=True,
                   ssh_host = 'green')
