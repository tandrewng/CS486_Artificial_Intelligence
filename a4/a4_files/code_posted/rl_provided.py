import numpy as np
import pickle
from typing import List, NewType, Tuple

State = NewType('State', Tuple[int, int])


# Note: This is only used for internal calculations. Keep using ints for the directions in your code.
# The coordinates if we move up, right, down, or left from the state (0,0).
_DIR_FOR_ACTION = [(-1, 0), (0, 1), (1, 0), (0, -1)]


class World:
    STATE_START = 'S'  # the initial state
    STATE_WALL = 'X'  # a wall, which is a non-goal state.
    STATE_NORMAL = '*'  # A non-goal state that is not a wall.

    def __init__(self, name: str, grid: np.ndarray, reward: float, discount: float):
        """
        Create the World object
        """

        self.name = name

        # The grid
        # S is the initial state.
        # X is a wall.
        # * denotes any other non-goal state.
        self.grid = grid

        # the immediate reward of entering any non-goal state
        self.reward = reward

        self.num_actions = 4

        # the discount factor
        self.discount = discount

        # load pre-generated random moves
        file_name = "world_{}_run.pickle".format(name)
        with open(file_name, 'rb') as run_file:
            self.run = pickle.load(run_file)

        # set start_state and curr_state
        # a state is encoded as a tuple (i,j) 
        # where i is the row index from the top and j is the column index from the left.
        # Both indices start from 0.
        start = np.where(self.grid == 'S')
        self.start_state = (start[0][0], start[1][0])
        self.curr_state = self.start_state

    def make_move_det(self, dir_intended: int, n_sa) -> State:
        """
        Given the current state (self.curr_state) and the intended direction (dir_intended),
        make a move (using the pre-determined random outcomes) and returns the next state.

        The intended direction is encoded as follows:
            up: 0, right: 1, down: 2, left: 3.
        """

        curr_num = n_sa[self.curr_state][dir_intended]
        curr_state = self.run[self.curr_state][dir_intended][int(curr_num)]

        # if curr_state is a goal state, respawn at the start state
        if is_goal(self.grid, curr_state):
            self.curr_state = self.start_state
        else:
            self.curr_state = curr_state

        return curr_state


def read_world(name: str):
    """
    Given the name of the world, load the world info from the file
    and create and return the World object.
    """

    file = "world_{}.txt".format(name)

    try:
        with open(file) as file_world:

            # read in the grid
            grid = []
            size = file_world.readline().split(',')
            for num in range(int(size[0])):
                line = file_world.readline()
                grid.append(line.split())
            grid = np.array(grid)

            # read in other info
            discount = float(file_world.readline())
            reward = float(file_world.readline())

    except Exception as ex:
        print("Error when processing grid: {0}".format(ex))
        print("Grid formatted incorrectly, please reference input format.")

    # create and return World object
    ret_world = World(name, grid, reward, discount)
    return ret_world


def get_next_state(grid, curr_state: State, action: int) -> State:
    """
    Return the state we end up in if we take the given action (no randomness involved).
    The next state is the current state if the agent bumps into a wall or the grid boundary.
    """

    n_rows, n_cols = grid.shape

    # The coordinates if we take action in state
    new_x = curr_state[0] + _DIR_FOR_ACTION[action][0]
    new_y = curr_state[1] + _DIR_FOR_ACTION[action][1]

    # Bounce back if out of grid boundary
    new_x = 0 if new_x < 0 else min(new_x, n_rows-1)
    new_y = 0 if new_y < 0 else min(new_y, n_cols-1)

    # Bounce back if in wall
    new_s = (new_x, new_y)
    new_s = new_s if not is_wall(grid, new_s) else curr_state

    return new_s


def get_next_states(grid, curr_state: State) -> List:
    """
    Given a grid the current state, return the next states for all four actions (no randomness involved).
    The returned list contains the next states for the actions: up, right, down, 
    and left, in this order. For example, the third element of the list is the
    next state if the agent ends up moving down from the current state. 

    The next state is the current state if the agent bumps into a wall or
    the boundary of the grid.
    """

    return [get_next_state(grid, curr_state, a) for a in range(4)]


def not_goal_nor_wall(grid, state: State) -> bool:
    """
    Returns true if the given state is not a goal state and nor a wall
    and returns false otherwise.
    """
    return grid[state] == World.STATE_NORMAL or grid[state] == World.STATE_START


def is_wall(grid, state: State) -> bool:
    """
    Returns true if the given state is a wall and returns false otherwise.
    """
    return grid[state] == World.STATE_WALL


def is_goal(grid, state: State) -> bool:
    """
    Returns true if the given state is a goal state 
    and returns false otherwise.
    """
    return (not is_wall(grid, state)) and (not not_goal_nor_wall(grid, state))


def get_pretty_policy(grid, policy: np.ndarray) -> np.ndarray:
    """
    Return the policy in a readable format
    where the actions are printed as up, right, down, and left. 
    
    policy must be the same shape as grid.
    """

    pretty_directions = ['up', 'right', 'down', 'left']
    pp_policy = np.zeros(grid.shape, dtype='U5')

    for state in np.ndindex(grid.shape):
        if is_wall(grid, state) or is_goal(grid, state):
            pp_policy[state] = str(grid[state])
        else:
            pp_policy[state] = pretty_directions[policy[state]]
        pp_policy[state] = pp_policy[state].ljust(5)
    return pp_policy


def get_pretty_utilities(grid, utils: np.ndarray) -> np.ndarray:
    """
    Return the utilities in a readable format.
    utils must be the same shape as grid.
    """

    pretty_utils = np.zeros(grid.shape, dtype='U5')

    for state in np.ndindex(grid.shape):
        if is_wall(grid, state):
            pretty_utils[state] = str(grid[state])
        else:
            pretty_utils[state] = f'{utils[state]:.3f}'
        pretty_utils[state] = pretty_utils[state].ljust(6)

    return pretty_utils


def utils_converged(utils, next_utils) -> bool:
    """
    Given the utility values for two consecutive iterations (utils and next_utils),
    return true if the sum of the absolute changes in all utility values is less than 0.001,
    and return false otherwise.
    """
    return np.max(np.abs(utils - next_utils)) < 0.001
