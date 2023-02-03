from rl_provided import *
import numpy as np
from typing import Tuple, List
from copy import *


def get_transition_prob(n_sa, n_sas, curr_state: State, dir_intended: int, next_state: State) -> float:
    """
    Determine the transition probability based on counts in n_sa and n_sas'.
    curr_state is s. dir_intended is a. next_state is s'.

    @return: 0 if we haven't visited the state-action pair yet (i.e. N_sa = 0).
      Otherwise, return N_sas' / N_sa.
    """

    if n_sa[curr_state[0]][curr_state[1]][dir_intended] == 0:
        return 0
    else:
        return n_sas[curr_state[0]][curr_state[1]][dir_intended][next_state[0]][next_state[1]] \
            / n_sa[curr_state[0]][curr_state[1]][dir_intended]


def exp_utils(world, utils, n_sa, n_sas, curr_state: State) -> List[float]:
    """
    @return: The expected utility values for all four possible actions.
    i.e. calculates sum_s'( P(s' | s, a) * U(s')) for all four possible actions.

    The returned list contains the expected utilities for the actions up, right, down, and left,
    in this order.  For example, the third element of the array is the expected utility
    if the agent ends up going down from the current state.
    """

    next_states = get_next_states(world.grid, curr_state)
    retval = []

    for dir_intended in range(4):
        sum_utils = 0 
        bounce_back_counter = 0
        bounce_total = 0
        for next_state in next_states:
            util_val = get_transition_prob(n_sa, n_sas, curr_state, dir_intended, next_state) * utils[next_state]
            if curr_state == next_state:
                bounce_total += util_val
                bounce_back_counter += 1
            else:
                sum_utils += util_val
        if bounce_back_counter > 0: bounce_total /= bounce_back_counter
        sum_utils += bounce_total
        retval.append(sum_utils)
    return retval


def optimistic_exp_utils(world, utils, n_sa, n_sas, curr_state: State, n_e: int, r_plus: float) -> List[float]:
    """
    @return: The optimistic expected utility values for all four possible actions.
    i.e. calculates f ( sum_s'( P(s' | s, a) * U(s')), N(s, a) ) for all four possible actions.

    The returned list contains the optimistic expected utilities for the actions up, right, down, and left,
    in this order.  For example, the third element of the array is the optimistic expected utility
    if the agent ends up going down from the current state.
    """
    e_utils = exp_utils(world, utils, n_sa, n_sas, curr_state)
    utils = []
    for a in range(4):
        if n_sa[curr_state[0]][curr_state[1]][a] < n_e:
            utils.append(r_plus)
        else:
            utils.append(e_utils[a])
    return utils



def update_utils(world, utils, n_sa, n_sas, n_e: int, r_plus: float) -> np.ndarray:
    """
    Update the utility values via value iteration until they converge.
    Call `utils_converged` to check for convergence.
    @return: The updated utility values.
    @rtype: An `np.ndarray` of size `world.grid.shape` of type `float`.
    """

    reward = world.reward
    discount = world.discount
    next_utils = np.zeros(world.grid.shape, dtype=float)
    row_size, col_size = world.grid.shape
    while True:
        for row in range(row_size):
            for col in range(col_size):
                curr_state = State((row, col))
                if not not_goal_nor_wall(world.grid, curr_state):
                    next_utils[row][col] = utils[row][col]
                else:
                    next_utils[row][col] = reward + discount\
                        * max(optimistic_exp_utils(world, utils, n_sa, n_sas, curr_state, n_e, r_plus))
        if utils_converged(utils, next_utils):break
        utils = deepcopy(next_utils)
    return next_utils



def get_best_action(world, utils, n_sa, n_sas, curr_state: State, n_e: int, r_plus: float) -> int:
    """
    @return: The best action, based on the agent's current understanding of the world, to perform in `curr_state`.
    """
    return np.argmax(optimistic_exp_utils(world, utils, n_sa, n_sas, curr_state, n_e, r_plus))


def adpa_move(world, utils, n_sa, n_sas, curr_state: State, n_e: int, r_plus: float) -> Tuple[State, np.ndarray]:
    """
    Execute ADP for one move. This function performs the following steps.
        1. Choose best action based on the utility values (utils).
        2. Make a move by calling `make_move_det`.
        3. Update the counts.
        4. Update the utility values (utils) via value iteration.
        5. Return the new state and the new utilities.

    @return: The state the agent ends up in after performing what it thinks is the best action + the updated
      utilities after performing this action.
    @rtype: A tuple (next_state, next_utils), where:
     - next_utils is an `np.ndarray` of size `world.grid.shape` of type `float`
    """
    best_dir = get_best_action(world, utils, n_sa, n_sas, curr_state, n_e, r_plus)
    next_state = world.make_move_det(best_dir, n_sa)
    n_sa[curr_state[0]][curr_state[1]][best_dir] += 1
    n_sas[curr_state[0]][curr_state[1]][best_dir][next_state[0]][next_state[1]] += 1
    next_utils = update_utils(world, utils, n_sa, n_sas, n_e, r_plus)
    return next_state, next_utils


def utils_to_policy(world, utils, n_sa, n_sas) -> np.ndarray:
    """
    @return: The optimal policy derived from the given utility values.
    @rtype: An `np.ndarray` of size `world.grid.shape` of type `int`.
    """
    # Initialize the policy.
    policy = np.zeros(world.grid.shape, dtype=int)
    row_size, col_size = world.grid.shape
    for row in range(row_size):
        for col in range(col_size):
            curr_state = State((row, col))
            policy[row][col] = np.argmax(exp_utils(world, utils, n_sa, n_sas, curr_state))
    return policy



def is_done_exploring(n_sa, grid, n_e: int) -> bool:
    """
    @return: True when the agent has visited each state-action pair at least `n_e` times.
    """
    s_x, s_y, action = n_sa.shape
    for x in range(s_x):
        for y in range(s_y):
            for a in range(action):
                if n_sa[x][y][a] < n_e and not_goal_nor_wall(grid, State((x, y))):
                    return False
    return True


def adpa(world_name: str, n_e: int, r_plus: float) -> np.ndarray:
    """
    Perform active ADP. Runs a certain number of moves and returns the learned utilities and policy.
    Stops when the agent is done exploring the world and the utility values have converged.
    Call `utils_converged` to check for convergence.

    Note: By convention, our tests expect the utility of a "wall" state to be 0.

    @param world_name: The name of the world we wish to explore.
    @param n_e: The minimum number of times we wish to see each state-action pair.
    @param r_plus: The maximum reward we can expect to receive in any state.
    @return: The learned utilities.
    @rtype: An `np.ndarray` of size `world.grid.shape` of type `float`.
    """
    # Initialize the world
    world = read_world(world_name)
    grid = world.grid
    num_actions = world.num_actions

    # Initialize persistent variable
    utils = np.zeros(grid.shape)
    row_size, col_size = grid.shape

    # populate utils
    for row in range(0, row_size):
        for col in range(0, col_size):
            curr_state = State((row, col))
            if is_goal(world.grid, curr_state):
                utils[row][col] = world.grid[row][col]

    n_sa = np.zeros((*grid.shape, num_actions))
    n_sas = np.zeros((*grid.shape, num_actions, *grid.shape))

    while True:
        state, next_utils = adpa_move(world, utils, n_sa, n_sas, world.curr_state, n_e, r_plus)

        if is_done_exploring(n_sa, world.grid, n_e) and utils_converged(utils, next_utils):
            return next_utils
        utils = deepcopy(next_utils)
        