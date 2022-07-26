from doodad.easy_launch.python_function import run_experiment

# The goal here is test doodad functionality in this dummy file.
# Once doodad works, we can use it for the dqn-Atari.py (the main experiment).

def foo(doodad_config, variant):
    print("The learning rate is", variant['learning_rate'])
    print("You are", variant['parameter'])
    print("Save to", doodad_config.base_log_dir)
    # save outputs (e.g. logs, parameter snapshots, etc.) to
    # doodad_config.base_log_dir

if __name__ == "__main__":
    variant = dict(
        learning_rate=1e-4,
        parameter='awesome',
    )
    run_experiment(
        foo,
        exp_name='test',
        mode='local',
        variant=variant,
    )





