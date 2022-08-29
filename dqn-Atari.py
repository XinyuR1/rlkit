"""
Run DQN on Atari environments
Experiment 1: Train on Assault, Test on Assault
Experiment 2: Train on Space Invaders and Carnival, Test on Assault

Modified from https://github.com/rail-berkeley/rlkit/blob/master/examples/dqn_and_double_dqn.py
"""

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

from atari_kit.wrappers import *
from rlkit.torch.networks.custom import ConvNet2
from doodad.easy_launch.python_function import run_experiment
from rlkit.core import logger

def experiment(doodad_config, variant):

    print('doodad_config.base_log_dir: ', doodad_config.base_log_dir)
    name = variant["exp_name"]
    output_path = f'{doodad_config.base_log_dir}/{name}/{variant["mode"]}'

    setup_logger(name, variant=variant,
                 log_dir=output_path)

    # EXPERIMENT 1
    #expl_env = make_env(["Assault-v0"])
    #eval_env = make_env(["Assault-v0"])

    # EXPERIMENT 2
    expl_env = make_env(["Carnival-v0", "SpaceInvaders-v0"])
    eval_env = make_env(["Assault-v0"])

    print(f'ACTIONS FOR EXPL: {expl_env.action_space.n}')
    print(f'ACTIONS FOR EVAL: {eval_env.action_space.n}')

    expl_n_actions = expl_env.action_space.n

    # Deep Q-Networks with CNN policy
    qf = ConvNet2(expl_n_actions)
    target_qf = ConvNet2(expl_n_actions)

    # Criterion and policy for training and test environments
    qf_criterion = nn.MSELoss()
    eval_policy = ArgmaxDiscretePolicy(qf)
    expl_policy = PolicyWrappedWithExplorationStrategy(
        EpsilonGreedy(expl_env.action_space),
        eval_policy
    )

    # Path collectors for both environments
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        expl_policy,
    )

    # Apply DQN Algorithm for these two experiments
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
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="DQN",
        version="normal",
        #mode="here_no_doodad",
        #mode="local",
        #mode="local_docker",
        mode="ssh",
        replay_buffer_size=int(1E5), #originally 1E6
        algorithm_kwargs=dict(
            # Original num_epochs: 3000
            num_epochs=5000,
            # 5000 - 1000 - 1000 - 1000 - 1000 - 256
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=500,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99, 
            learning_rate=3E-4
        ),
        exp_name='DQN-CSI-A'
    )

    # If use GPU, uncomment the following line
    ptu.set_gpu_mode(True)

    # Run the experiment using cpu or gpu (from the lab computer)
    run_experiment(experiment, 
        exp_name=variant["exp_name"], 
        use_gpu=True,
        #use_gpu=False,
        ssh_host='blue',
        variant=variant, mode=variant["mode"]
    )

