# version 1.0

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from io import BytesIO

class Environment:

    def __init__(self, num_state_types: int, state_types: List[int], \
                       num_observe_types: int, observe_probs: List[Dict], \
                       action_effects: List[Dict], transition_matrices):
        '''
        Initialize a representation of the environment. 
        Set attributes of the environment and initialize the state at time step 0.

        :param num_state_types: Number of state types. 
        :param state_types: List of state types.

        :param num_observe_types: Number of observation types.
        :param observe_probs: List of Dicts describing the probabilities of 
            generating the observation types for each state type.

        :param action_effects: List of Dicts describing the effects of the actions.
        :param transition_matrices: A matrix representing the probabilities of 
            transitioning from one state to another.
        '''
        
        # Check input requirements
        for state_type in state_types:
            assert 0 <= state_type <= num_state_types - 1, "Each state type must be in [0, num_state_types - 1]"
        assert len(observe_probs) == num_state_types, "Length of observe_probs should be equal to the number of state types"
        for probs in observe_probs:
            assert len(probs) == num_observe_types, "Length of each element in observe_probs should be equal to the number of observation types"

        self.num_state_types = num_state_types 
        self.state_types = state_types
        self.num_observe_types = num_observe_types

        # infer # of states from the state_types list
        self.num_states = len(self.state_types) 

        # We will use this to generate the observation matrix
        self.observe_probs = observe_probs
        
        # We will use these to generate the transition matrices
        self.action_effects = action_effects
        self.transition_matrices = transition_matrices
        if self.action_effects is not None: 
            self.num_actions = len(action_effects)
        else: 
            self.num_actions = len(transition_matrices)
        
        #======================================================================
        # Initialize the state for time 0 randomly
        self.__pos = np.random.randint(0, self.num_states - 1)  

        # Keep track of the sequence of states in the past
        self.__trajectory = [self.__pos]                    

    #==========================================================================
    # You do not need the functions below for the assignment.
    # However, you might find them helpful for simulating the environment.
    #==========================================================================
    def observe(self) -> int:
        '''
        Returns an observation at the current time step.
        :return: An observation at the current time step.
        '''
        state_type = self.state_types[self.__pos]
        distribution = self.observe_probs[state_type]

        observation = np.random.choice(range(len(distribution)), p=distribution)
        return observation
        
    def move(self, action: int):
        '''
        Simulate the action according to the transition probabilities for the action
        '''
        distribution = self.action_effects[action]
        state_offset = np.random.choice(list(distribution.keys()), p=list(distribution.values()))

        self.__pos = (self.__pos + state_offset) % self.num_states
        self.__trajectory.append(self.__pos)

    def get_cur_pos(self) -> int:
        '''
        Returns the current state
        :return: The current state
        '''
        return self.__pos

    def get_past_pos(self, k: int) -> int:
        '''
        Returns the state at time step k in the past
        :return: The state at time step k in the past
        '''
        assert k < len(self.__trajectory), "k must be less than the number of past positions."
        return self.__trajectory[k]

    def act_and_observe(self, actions: List[int]) -> List[int]:
        '''
        Simulate the effect of taking the provided list of actions in the environment, 
        returning a list of observations.
        An initial observation is collected before actions are applied. 
        If you pass a list of n actions, a list of n + 1 observations are returned.

        :param actions: A list of actions. 
                        For example: [1, 1, 2, 0]
        :return: A list of observations.
                 For example: [0, 1, 1, 1, 1]
        '''

        assert all(a in range(self.num_actions) for a in actions), "One or more actions are invalid."

        # Alternate between collecting an observation and taking an action
        observations = []
        observations.append(self.observe())  # Collect observation in initial state

        # Apply each action, collecting an observation after each transition
        for a in actions:
            self.move(a) # Apply the action
            observations.append(self.observe())   # Collect an observation
        return observations



def visualize_belief(env: Environment, probs: Dict, k=None):
    '''
    Plot the current state and the agent's probabilistic beliefs 
    over the states at time k.

    Yellow bars are the states of type 0. 
    The red rectangle is the state that the agent is in at time step k.

    :param env: The environment.
    :param probs: The agent's beliefs over the states at time k.
    :param k: The time step at which to plot the state. 
    '''
    fig, ax = plt.subplots(2, gridspec_kw={'height_ratios': [3, 1]})

    # Plot the agent's beliefs for the state at time k.
    locs = list(range(env.num_states))
    ax[0].bar(locs, probs)

    ax[0].set_xlabel('Locations')
    ax[0].set_ylabel('Location Belief Probabilities')

    ax[0].set_xticks(np.arange(0, env.num_states, 1))
    ax[0].set_ylim([0., 1.])

    # Plot the states of type 0
    states_type_zero = np.zeros((env.num_states))
    for i in range(env.num_states):
        if env.state_types[i] == 0:
            states_type_zero[i] = 1.
    ax[1].bar(locs, states_type_zero, color='yyyy', label='states of type 0')

    # Plot the state that the agent is in at time step k.
    curr_state = env.get_cur_pos() if k is None else env.get_past_pos(k)
    ax[1].bar(curr_state, 0.5, color='r', label='current state')
    ax[1].legend(prop={"size": 8})

    ax[1].set_xlabel('Locations')
    ax[1].set_ylim([0., 2.])

    ax[1].set_xticks(np.arange(0, env.num_states, 1))
    ax[1].set_yticks([])
    
    plt.tight_layout()
    plt.show()


# class PMF:
#     '''
#     A class representing a normalized probability mass function over a discrete random variable.
#     '''

#     def __init__(self, domain: List, probabilities: List[float]):
#         '''
#         :param domain: A list of values for the random variable
#         :param probabilities: A list of probabilities corresponding to each value of the random variable
#         '''

#         assert (len(domain) == len(probabilities)), "Number of variables in domain must match number of probabilities"
#         self.domain = np.array(domain)
#         self.n_vars = len(domain)
#         self.probabilities = np.array(probabilities, dtype=float)

#         # normalize the probabilities
#         self.probabilities /= np.sum(self.probabilities)


#     def sample(self):
#         '''
#         Sample a value of the random variable from its probability distribution
#         :return: The sampled value of the random variable
#         '''
#         return np.random.choice(self.domain, p=self.probabilities)

#     def normalize(self):
#         '''
#         Normalize the probability distribution
#         '''
#         self.probabilities /= np.sum(self.probabilities)

#     def __getitem__(self, value):
#         '''
#         Return the probability associated with a particular value of the random variable.
#         Can also be accessed using PMF[value]
#         :param value: A value of the random variable (must exist in self.domain)
#         :return: The probability associated with that value
#         '''
#         if value in self.domain:
#             return self.probabilities[self.domain.tolist().index(value)]
#         else:
#             return 0.

