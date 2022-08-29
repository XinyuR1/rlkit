import gym
# import roboverse
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.data_management.split_buffer import SplitReplayBuffer
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.launchers.launcher_util import setup_logger
from rlkit.policies.argmax import ArgmaxDiscretePolicy
from rlkit.torch.networks.custom import ConvNet1, ConvNet2
from rlkit.torch.networks.mlp import Mlp
#from rlkit.envs.wrappers import NormalizedBoxEnv, StackObservationEnv, RewardWrapperEnv
import rlkit.torch.pytorch_util as ptu
from rlkit.samplers.data_collector import MdpPathCollector, ObsDictPathCollector
from rlkit.samplers.data_collector.step_collector import MdpStepCollector
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.awac_trainer import AWACTrainer
from rlkit.torch.torch_rl_algorithm import (
    TorchBatchRLAlgorithm,
    TorchOnlineRLAlgorithm,
)

from rlkit.demos.source.hdf5_path_loader import HDF5PathLoader
from rlkit.demos.source.mdp_path_loader import MDPPathLoader
#from rlkit.visualization.video import save_paths, VideoSaveFunction

#from multiworld.core.flat_goal_env import FlatGoalEnv
#from multiworld.core.image_env import ImageEnv
#from multiworld.core.gym_to_multi_env import GymToMultiEnv
from rlkit.util.hyperparameter import recursive_dictionary_update

import torch
from torch.nn import functional as F
import numpy as np
from torchvision.utils import save_image

from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_and_epsilon_strategy import GaussianAndEpsilonStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy

import os.path as osp
from rlkit.core import logger
from rlkit.util.io import load_local_or_remote_file
import pickle

from torch import nn as nn
from atari_kit.wrappers import *

from rlkit.torch.networks import LinearTransform

import random


ENV_PARAMS = {
    'HalfCheetah-v2': {
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_demo_path': dict(
            path="demos/icml2020/mujoco/hc_action_noise_15.npy",
            obs_dict=False,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            path="demos/icml2020/mujoco/hc_off_policy_15_demos_100.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'Ant-v2': {
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_demo_path': dict(
            path="demos/icml2020/mujoco/ant_action_noise_15.npy",
            obs_dict=False,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            path="demos/icml2020/mujoco/ant_off_policy_15_demos_100.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'Walker2d-v2': {
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_demo_path': dict(
            path="demos/icml2020/mujoco/walker_action_noise_15.npy",
            obs_dict=False,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            path="demos/icml2020/mujoco/walker_off_policy_15_demos_100.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },

    'SawyerRigGrasp-v0': {
        'env_id': 'SawyerRigGrasp-v0',
        # 'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 50,
        # 'num_epochs': 1000,
    },

    'pen-binary-v0': {
        'env_id': 'pen-binary-v0',
        'max_path_length': 200,
        'sparse_reward': True,
        'env_demo_path': dict(
            path="demos/icml2020/hand/pen2_sparse.npy",
            # path="demos/icml2020/hand/sparsity/railrl_pen-binary-v0_demos.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            # path="demos/icml2020/hand/pen_bc_sparse1.npy",
            # path="demos/icml2020/hand/pen_bc_sparse2.npy",
            # path="demos/icml2020/hand/pen_bc_sparse3.npy",
            # path="demos/icml2020/hand/pen_bc_sparse4.npy",
            path="demos/icml2020/hand/pen_bc_sparse4.npy",
            # path="ashvin/icml2020/hand/sparsity/bc/pen-binary1/run10/id*/video_*_*.p",
            # sync_dir="ashvin/icml2020/hand/sparsity/bc/pen-binary1/run10",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'door-binary-v0': {
        'env_id': 'door-binary-v0',
        'max_path_length': 200,
        'sparse_reward': True,
        'env_demo_path': dict(
            path="demos/icml2020/hand/door2_sparse.npy",
            # path="demos/icml2020/hand/sparsity/railrl_door-binary-v0_demos.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            # path="demos/icml2020/hand/door_bc_sparse1.npy",
            # path="demos/icml2020/hand/door_bc_sparse3.npy",
            path="demos/icml2020/hand/door_bc_sparse4.npy",
            # path="ashvin/icml2020/hand/sparsity/bc/door-binary1/run10/id*/video_*_*.p",
            # sync_dir="ashvin/icml2020/hand/sparsity/bc/door-binary1/run10",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'relocate-binary-v0': {
        'env_id': 'relocate-binary-v0',
        'max_path_length': 200,
        'sparse_reward': True,
        'env_demo_path': dict(
            path="demos/icml2020/hand/relocate2_sparse.npy",
            # path="demos/icml2020/hand/sparsity/railrl_relocate-binary-v0_demos.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            # path="demos/icml2020/hand/relocate_bc_sparse1.npy",
            path="demos/icml2020/hand/relocate_bc_sparse4.npy",
            # path="ashvin/icml2020/hand/sparsity/bc/relocate-binary1/run10/id*/video_*_*.p",
            # sync_dir="ashvin/icml2020/hand/sparsity/bc/relocate-binary1/run10",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    }
}


def resume(variant):
    data = load_local_or_remote_file(variant.get("pretrained_algorithm_path"), map_location="cuda")
    algo = data['algorithm']

    algo.num_epochs = variant['num_epochs']

    post_pretrain_hyperparams = variant["trainer_kwargs"].get("post_pretrain_hyperparams", {})
    algo.trainer.set_algorithm_weights(**post_pretrain_hyperparams)

    algo.train()

def process_args(variant):
    if variant.get("debug", False):
        variant['max_path_length'] = 50
        variant.get('algo_kwargs', {}).update(dict(
            batch_size=5,
            start_epoch=-2, # offline epochs
            num_epochs=2, # online epochs
            num_eval_steps_per_epoch=100,
            num_expl_steps_per_train_loop=100,
            num_trains_per_train_loop=10,
            min_num_steps_before_training=10,
        ))

    env_id = variant.get("env_id", None)
    if env_id:
        env_params = ENV_PARAMS.get(env_id, {})
        recursive_dictionary_update(variant, env_params)

def split_into_trajectories(replay_buffer):
    dones_float = np.zeros_like(replay_buffer._rewards)

    for i in range(replay_buffer._size):
        delta = replay_buffer._observations[i + 1, :] - replay_buffer._next_obs[i, :]
        norm = np.linalg.norm(delta)
        if norm > 1e-6 or replay_buffer._terminals[i]:
            dones_float[i] = 1
        else:
            dones_float[i] = 0

    trajs = [[]]

    for i in range(replay_buffer._size):
        trajs[-1].append((replay_buffer._observations[i],
            replay_buffer._actions[i],
            replay_buffer._rewards[i],
            replay_buffer._terminals[i],
            replay_buffer._next_obs[i]))
        if dones_float[i] == 1.0 and i + 1 < replay_buffer._size:
            trajs.append([])

    return trajs


def get_normalization(replay_buffer):
    trajs = split_into_trajectories(replay_buffer)

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    reward_range = compute_returns(trajs[-1]) - compute_returns(trajs[0])
    m = 1000.0 / reward_range
    return LinearTransform(m=float(m), b=0)

def experiment(doodad_config, variant):
    
    print('doodad_config.base_log_dir: ', doodad_config.base_log_dir)
    name = variant["exp_name"]
    output_path = f'{doodad_config.base_log_dir}/{name}/{variant["mode"]}'

    setup_logger(name, variant=variant,
                    log_dir=output_path)

    expl_env = make_env(variant["expl_env"])
    eval_env = make_env(variant["eval_env"])

    seed = int(variant["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    eval_env.seed(seed)
    expl_env.seed(seed)

    obs_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.n

    qf = ConvNet2(action_dim)
    target_qf = ConvNet2(action_dim)
    qf_criterion = nn.MSELoss()

    # EVALUATION 
    eval_policy = ArgmaxDiscretePolicy(qf)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )

    # EXPLORATION
    expl_policy = PolicyWrappedWithExplorationStrategy(
        EpsilonGreedy(expl_env.action_space),
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        expl_policy,
    )
    
    # REPLAY BUFFER
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )

    # TRAINER CLASS: IQL HERE
    trainer_class = variant.get("trainer_class", AWACTrainer)
    trainer = trainer_class(
        qf = qf,
        target_qf = target_qf,
        qf_criterion = qf_criterion,
        **variant['trainer_kwargs']
    )

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        max_path_length=variant['max_path_length'],
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()
    

    
