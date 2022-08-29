"""Torch implementation of Implicit Q-Learning (IQL)
https://github.com/ikostrikov/implicit_q_learning
"""

import pickle
from collections import OrderedDict
import numpy as np
import torch
import torch.optim as optim
from rlkit.torch.sac.policies import MakeDeterministic
from torch import nn as nn
import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.core import logger
from rlkit.core.logging import add_prefix
from rlkit.util.ml_util import PiecewiseLinearSchedule, ConstantSchedule
import torch.nn.functional as F
from rlkit.torch.networks import LinearTransform
import time


class IQLTrainer(TorchTrainer):
    def __init__(
            self,
            qf,
            target_qf,
            qf_criterion = None,
            learning_rate = 1e-3,
            quantile=0.5,
            buffer_policy=None,
            z=None,

            discount=0.99,
            reward_scale=1.0,

            policy_weight_decay=0,
            q_weight_decay=0,
            optimizer_class=optim.Adam,

            policy_update_period=1,
            q_update_period=1,

            reward_transform_class=None,
            reward_transform_kwargs=None,
            terminal_transform_class=None,
            terminal_transform_kwargs=None,

            clip_score=None,
            soft_target_tau=1e-2,
            target_update_period=1,
            beta=1.0,
    ):
        super().__init__()
        
        #Updated for DQN
        self.qf = qf
        self.target_qf = target_qf
        self.qf_criterion = qf_criterion
        self.learning_rate = learning_rate
        
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        
        self.z = z
        self.buffer_policy = buffer_policy

        self.optimizers = {}

        self.qf_optimizer = optimizer_class(
            self.qf.parameters(),
            weight_decay=q_weight_decay,
            lr=learning_rate,
        )

        if self.z:
            self.z_optimizer = optimizer_class(
                self.z.parameters(),
                weight_decay=q_weight_decay,
                lr=learning_rate, #initially qf_lr
            )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        self.q_update_period = q_update_period
        self.policy_update_period = policy_update_period

        self.reward_transform_class = reward_transform_class or LinearTransform
        self.reward_transform_kwargs = reward_transform_kwargs or dict(m=1, b=0)
        self.terminal_transform_class = terminal_transform_class or LinearTransform
        self.terminal_transform_kwargs = terminal_transform_kwargs or dict(m=1, b=0)
        self.reward_transform = self.reward_transform_class(**self.reward_transform_kwargs)
        self.terminal_transform = self.terminal_transform_class(**self.terminal_transform_kwargs)

        self.clip_score = clip_score
        self.beta = beta
        self.quantile = quantile

    def train_from_torch(self, batch, train=True, pretrain=False,):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        if self.reward_transform:
            rewards = self.reward_transform(rewards)

        if self.terminal_transform:
            terminals = self.terminal_transform(terminals)
        """
        Policy and Alpha Loss
        """
        #dist = self.policy(obs)

        """
        QF Loss
        """
        target_q_values = self.target_qf(next_obs).detach().max(1, keepdim = True)[0]
        q_target = rewards + (1. - terminals) * self.discount * target_q_values
        q_target = q_target.detach()
        # actions is a one-hot vector
        q_pred = torch.sum(self.qf(obs) * actions, dim=1, keepdim=True)
        qf_loss = self.qf_criterion(q_pred, q_target)
    
        """
        Update networks
        """
        if self._n_train_steps_total % self.q_update_period == 0:
            
            self.qf_optimizer.zero_grad()
            qf_loss.backward()
            self.qf_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.qf, self.target_qf, self.soft_target_tau
            )

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'rewards',
                ptu.get_numpy(rewards),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'terminals',
                ptu.get_numpy(terminals),
            ))
            
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        nets = [
            self.qf,
            self.target_qf,
        ]
        return nets

    def get_snapshot(self):
        return dict(
            qf = self.qf,
            target_qf = self.target_qf,
        )
