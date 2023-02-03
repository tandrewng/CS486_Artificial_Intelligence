import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from io import BytesIO

from utils_soln import *
from fba import *

# num_state_types = 2
# state_types = [1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]
# num_observe_types = 2
# observe_probs = [[0.8, 0.2], [0.1, 0.9]]
# action_effects = [{0: 0.1, -1: 0.8, -2: 0.1}, {0: 0.1, 1: 0.8, 2: 0.1}, {0: 1.0}]
# transition_matrices = [[[0.7, 0.3], [0.3, 0.7]]]
# test_env = Environment(num_state_types, state_types, num_observe_types,observe_probs, \
#     action_effects, transition_matrices)

# test_env2 = Environment(num_state_types, state_types, num_observe_types,observe_probs, \
#     action_effects, None)

# # create_observation_matrix(test_env)

# # print(test_env.observe_matrix)

# create_transition_matrices(test_env2)
# print(test_env2.transition_matrices)

# num_state_types = 2
# state_types = [0, 1]
# num_observe_types = 2
# observe_probs = [[0.1, 0.9], [0.8, 0.2]]
# action_effects = [{0: 0.05, -1: 0.95}, {0: 0.15, 1: 0.85}, {0: 1.0}]
# test_env3 = Environment(num_state_types, state_types, num_observe_types,observe_probs, action_effects, None)

# # arr = forward_recursion(test_env3, [1, 0], [1, 0, 0], [0.5, 0.5])
# # print(arr)
# arr = fba(test_env3, [1, 0], [1, 0, 0], [0.5, 0.5])
# print(arr)


# num_state_types = 2
# state_types = [0, 1]
# num_observe_types = 2
# observe_probs = [[0.9, 0.1], [0.2, 0.8]]
# transition_matrices = [[[0.7, 0.3], [0.3, 0.7]]]
# action_effects = [{0: 0.05, -1: 0.95}, {0: 0.15, 1: 0.85}, {0: 1.0}]
# test_env3 = Environment(num_state_types, state_types, num_observe_types,observe_probs, None, transition_matrices)

# arr = forward_recursion(test_env3, [1, 0], [1, 0, 0], [0.5, 0.5])
# print(arr)
# arr = backward_recursion(test_env3, [0], [0, 0])
# arr = fba(test_env3, [], [0, 0], [0.5, 0.5])
# print(arr)

num_state_types = 2
state_types = [0, 1]
num_observe_types = 2
observe_probs = [[0.1, 0.9], [0.8, 0.2]]
action_effects = [{0: 0.05, -1: 0.95}, {0: 0.15, 1: 0.85}, {0: 1.0}]
test_env3 = Environment(num_state_types, state_types, num_observe_types,observe_probs, action_effects, None)

# arr = forward_recursion(test_env3, [1, 0], [1, 0, 0], [0.5, 0.5])
# print(arr)
arr = fba(test_env3, [1, 0], [0, 1, 1], [0.5, 0.5])
print(arr)