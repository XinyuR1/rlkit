import numbers
import random

from rlkit.exploration_strategies.base import RawExplorationStrategy

number = 0
epsilon = 1

class EpsilonGreedy(RawExplorationStrategy):
    """
    Take a random discrete action with some probability.
    """
    def __init__(self, action_space, prob_random_action = 0.1):
        self.prob_random_action = prob_random_action
        self.action_space = action_space

    
    def get_action_from_raw_action(self, action, **kwargs):
        global number
        global epsilon
        number += 1
        threshold = 1000000

        # The value of epsilon will decrease from 1 to 0.1 in
        # 1 million steps (threshold)
        if number <= threshold:
            epsilon -= (1-0.1)/threshold
        
        if number % 1000 == 0:
            print(f'NUMBER: {number}')
            print(f'EPSILON VALUE: {epsilon}')

        if random.random() <= epsilon:
            return self.action_space.sample()
        return action
