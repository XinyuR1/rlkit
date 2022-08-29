"""
Run IQL on Atari environments
Experiment 3: Train on Assault, Test on Assault
Experiment 4: Train on Space Invaders and Carnival, Test on Assault

Modified from: https://github.com/rail-berkeley/rlkit/blob/master/examples/iql/antmaze_finetune.py 
"""

from rlkit.launchers.experiments.awac.finetune_rl import experiment
from doodad.easy_launch.python_function import run_experiment
from rlkit.torch.dqn.iql_trainer import IQLTrainer
import rlkit.torch.pytorch_util as ptu
import random

variant = dict(
    # EXPERIMENT 3
    expl_env = ["Assault-v0"],
    eval_env = ["Assault-v0"],

    # EXPERIMENT 4
    #expl_env = ["SpaceInvaders-v0", "Carnival-v0"],
    #eval_env = ["Assault-v0"],


    algo_kwargs=dict(
        start_epoch=-1000, # offline epochs
        num_epochs=4000, # online epochs (4000)
        batch_size=256,
        num_eval_steps_per_epoch=1000,
        num_trains_per_train_loop=1000,
        num_expl_steps_per_train_loop=1000,
        min_num_steps_before_training=1000,
    ),
    replay_buffer_size=int(1E5), #originally 2e-6
    max_path_length=500,
    layer_size=256,
    
    # trainer
    algorithm="DQN",
    version="normal",
    collection_mode='batch',
    #mode = 'here_no_doodad',
    #mode='local',
    #mode = 'local_docker',
    mode = 'ssh',
    trainer_class=IQLTrainer,
    trainer_kwargs=dict(
        discount=0.99,
        reward_scale=1,

        q_weight_decay=0,

        reward_transform_kwargs=dict(m=1, b=-1),
        terminal_transform_kwargs=None,

        beta=0.1,
        quantile=0.9,
        clip_score=100,
    ),

    seed=random.randint(0, 100000),
    exp_name = 'IQL-CSI-A'
)

def main():
    # If use GPU, uncomment the following line
    ptu.set_gpu_mode(True)

    run_experiment(experiment,
        variant=variant,
        exp_name=variant["exp_name"],
        mode=variant["mode"],
        ssh_host='blue',
        use_gpu=True,
        #use_gpu = False,
    )

if __name__ == "__main__":
    main()



