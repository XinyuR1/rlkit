import glob
import os
import sys

def get_choice_env():
    env_choice = "Select which Atari environment you want to work with:\n" \
                 "(1): Breakout-v0\n" \
                 "(2): Pong-v0\n" \
                 "(3): BeamRider-v0\n" \
                 "(4): Seaquest-v0\n" \
                 "\nEnter the number: "
    env_number = input(env_choice)
    env_name = None

    if int(env_number) == 1:
        env_name = "Breakout-v0"
    elif int(env_number) == 2:
        env_name = "Pong-v0"
    elif int(env_number) == 3:
        env_name = "BeamRider-v0"
    elif int(env_number) == 4:
        env_name = "Seaquest-v0"
    else:
        print('Invalid Atari environment for this experiment.')
        exit()

    print(f'Chosen Atari Environment: {env_name}\n')
    return env_name

"""
Testing
if __name__ == "__main__":
    print(get_choice_env())
"""

