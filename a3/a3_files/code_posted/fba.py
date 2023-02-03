# version 1.0

import numpy as np
from typing import List, Dict

from utils_soln import *

def create_observation_matrix(env: Environment):
    '''
    Creates a 2D numpy array containing the observation probabilities for each state. 

    Entry (i,j) in the array is the probability of making an observation type j in state i.

    Saves the matrix in env.observe_matrix and returns nothing.
    '''

    #### Your Code Here ####
    env.observe_matrix = np.zeros((env.num_states, env.num_observe_types))
    for i in range(0, env.num_states):
        for j in range(0, env.num_observe_types):
            env.observe_matrix[i][j] = env.observe_probs[env.state_types[i]][j]
    return


def create_transition_matrices(env: Environment):
    '''
    If the transition_matrices in env is not None, 
    constructs a 3D numpy array containing the transition matrix for each action.

    Entry (i,j,k) is the probability of transitioning from state j to k
    given that the agent takes action i.

    Saves the matrices in env.transition_matrices and returns nothing.
    '''

    if env.transition_matrices is not None:
        return

    #### Your Code Here ####
    env.transition_matrices = np.zeros((env.num_actions, env.num_states, env.num_states))
    for i in range(0, env.num_actions):
        for j in range(0, env.num_states):
            for key, val in env.action_effects[i].items():
                k = (j + key) % env.num_states
                env.transition_matrices[i][j][k] = val
    return

def forward_recursion_helper(env: Environment, action: int, obs: int, \
    prev: List[float], probs_init: List[float] = None):

    ret_val = []
    if (probs_init is not None):
        for i in range(0, env.num_states):
            f_00 = env.observe_matrix[i][obs]*probs_init[i]
            ret_val.append(f_00)

        # normalize
        alpha = sum(ret_val)
        for i in range(0, len(ret_val)):
            ret_val[i] /= alpha
    else:
        for s_k in range(0, env.num_states):
            P_Sk_sk1 = 0
            for s_k_1 in range(0, env.num_states):
                P_Sk_sk1 += env.transition_matrices[action][s_k_1][s_k] * prev[s_k_1] * \
                    env.observe_matrix[s_k][obs]
            ret_val.append(P_Sk_sk1)
        
        # normalize
        alpha = sum(ret_val)
        for i in range(0, len(ret_val)):
            ret_val[i] /= alpha
    return ret_val
    


def forward_recursion(env: Environment, actions: List[int], observ: List[int], \
    probs_init: List[float]) -> np.ndarray:
    '''
    Perform the filtering task for all the time steps.

    Calculate and return the values f_{0:0} to f_{0:t-1} where t = len(observ).

    :param env: The environment.
    :param actions: The actions for time steps 0 to t - 2.
    :param observ: The observations for time steps 0 to t - 1.
    :param probs_init: The initial probabilities over the N states.

    :return: A numpy array with shape (t, N) (N is the number of states)
        the k'th row represents the normalized values of f_{0:k} (0 <= k <= t - 1).
    '''
    ### YOUR CODE HERE ###
    create_observation_matrix(env)
    create_transition_matrices(env)
    t = len(observ)
    ret_val = np.zeros((t, env.num_states))
    ret_val[0] = forward_recursion_helper(env, None, observ[0], None, probs_init)

    for k in range(1, t):
        if not actions: 
            action = 0
        else:
            action = actions[k-1]
        ret_val[k] = forward_recursion_helper(env, action, observ[k], ret_val[k-1])
    return ret_val

def backward_recursion_helper(env: Environment, action: int, obs: int, \
    prev: List[float]):
    ret_val = []
    for s_k in range(0, env.num_states):
            P_Sk_sk1 = 0
            for s_k_1 in range(0, env.num_states):
                P_Sk_sk1 += env.transition_matrices[action][s_k_1][s_k] * prev[s_k_1] * \
                    env.observe_matrix[s_k_1][obs]
            ret_val.append(P_Sk_sk1)
    
    return ret_val

def backward_recursion(env: Environment, actions: List[int], observ: List[int] \
    ) -> np.ndarray:
    '''
    Perform the smoothing task for each time step.

    Calculate and return the values b_{1:t-1} to b_{t:t-1} where t = len(observ).

    :param env: The environment.
    :param actions: The actions for time steps 0 to t - 2.
    :param observ: The observations for time steps 0 to t - 1.

    :return: A numpy array with shape (t+1, N), (N is the number of states)
            the k'th row represents the values of b_{k:t-1} (1 <= k <= t - 1),
            while the k=0 row is meaningless and we will NOT test it.
    '''

    ### YOUR CODE HERE ###
    create_observation_matrix(env)
    create_transition_matrices(env)
    t = len(observ)
    ret_val = np.zeros((t + 1, env.num_states))
    ret_val[t] = np.ones(env.num_states)

    for k in range(t-1, 0, -1):
        if not actions: 
            action = 0
        else:
            action = actions[k-1]
        ret_val[k] = backward_recursion_helper(env, action, observ[k] , ret_val[k+1])
    return ret_val


def fba(env: Environment, actions: List[int], observ: List[int], \
    probs_init: List[float]) -> np.ndarray:
    '''
    Execute the forward-backward algorithm. 

    Calculate and return a 2D numpy array with shape (t,N) where t = len(observ) and N is the number of states.
    The k'th row represents the smoothed probability distribution over all the states at time step k.

    :param env: The environment.
    :param actions: A list of agent's past actions.
    :param observ: A list of observations.
    :param probs_init: The agent's initial beliefs over states
    :return: A numpy array with shape (t, N)
        the k'th row represents the normalized smoothed probability distribution over all the states for time k.
    '''

    ### YOUR CODE HERE ###
    t = len(observ)
    create_observation_matrix(env)
    create_transition_matrices(env)
    f = forward_recursion(env, actions, observ, probs_init)
    b = backward_recursion(env, actions, observ)
    ret_val = np.zeros((t, env.num_states))
    for i in range(0, t):
        for j in range(0, env.num_states):
            ret_val[i][j] = f[i][j] * b[i + 1][j]
    # print(f)
    # print(b)
    # normalize
    for i in range(0, len(ret_val)):
        alpha = sum(ret_val[i])
        for j in range(0, len(ret_val[i])):
            ret_val[i][j] /= alpha
    return ret_val

