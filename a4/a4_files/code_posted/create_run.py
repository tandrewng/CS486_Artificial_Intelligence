import argparse
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, NewType
from math import cos, sin, radians, isnan
import pickle

State = NewType('State', Tuple[int, int])
Action = NewType('Action', Tuple[int, int])

N = Action((-1, 0))
E = Action((0, 1))
S = Action((1, 0))
W = Action((0, -1))


@dataclass
class MDP:
    grid: List[List[float]]
    init_state: State
    terminal_states: List[State]
    actions: List[Action] = field(default_factory=lambda: [N, E, S, W])
    states: List[State] = field(default_factory=list)
    rewards: Dict[State, float] = field(default_factory=dict)
    transition_probs: Dict[float, float] = field(default_factory=dict)  # Dict[rotation in degrees, transition prob]
    transition_table: Dict[Tuple[State, Action], Dict[State, float]] = field(default_factory=dict)
    discount: float = 1.0
    n_rows: int = 0
    n_cols: int = 0

    def move_deterministically(self, cur_state: State, action: Action) -> State:
        if cur_state not in self.states:
            raise Exception("Invalid state.")
        if action not in self.actions:
            raise Exception("Invalid action.")

        new_x = cur_state[0] + action[0]
        new_y = cur_state[1] + action[1]
        new_x = 0 if new_x < 0 else min(new_x, self.n_cols)
        new_y = 0 if new_x < 0 else min(new_y, self.n_rows)
        new_s = State((new_x, new_y))
        new_s = new_s if new_s in self.states else cur_state
        return new_s

    def is_terminal(self, cur_state: State) -> bool:
        if cur_state not in self.states:
            raise Exception("Invalid state.")
        return cur_state in self.terminal_states

    def _populate_states(self):
        self.n_rows = len(self.grid)
        self.n_cols = len(self.grid[0])
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if not isnan(self.grid[r][c]) and self.grid[r][c]:
                    s = State((r, c))
                    self.states.append(s)
                    self.rewards[s] = self.grid[r][c]

    def _populate_transition_probs(self):
        for s in self.states:
            for a in self.actions:
                self.transition_table[(s, a)] = {}
                if not self.is_terminal(s):
                    for rot_deg, prob in self.transition_probs.items():
                        if prob != 0:
                            rotated_action = _rotate(a, rot_deg)
                            new_s = self.move_deterministically(s, rotated_action)
                            if new_s in self.transition_table[(s, a)]:
                                self.transition_table[(s, a)][new_s] += prob
                            else:
                                self.transition_table[(s, a)][new_s] = prob

    def __post_init__(self):
        self._populate_states()
        if self.transition_probs:
            self._populate_transition_probs()


def _rotate(facing_direction: Action, rotation_degree: float) -> Action:
    a, deg = facing_direction, radians(-rotation_degree)
    dx = a[0] * cos(deg) - a[1] * sin(deg)
    dy = a[0] * sin(deg) + a[1] * cos(deg)
    return Action((int(dx), int(dy)))


def read_secret_world(name: str):
    file = "world_{}_secret.txt".format(name)

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
            transition_probs = map(float, file_world.readline().split(','))

    except Exception as ex:
        print("Error when processing grid: {0}".format(ex))
        print("Grid formatted incorrectly, please reference input format.")

    # Reformat to work with the MDP class
    init_state = ()
    terminal_states = []
    new_grid = np.zeros(grid.shape)
    for state in np.ndindex(grid.shape):
        if grid[state].lstrip('-+').isnumeric():
            terminal_states.append(state)
            new_grid[state] = float(grid[state])
        elif grid[state] == 'S':
            init_state = state
            new_grid[state] = reward
        elif grid[state] == 'X':
            new_grid[state] = None
        else:
            new_grid[state] = reward

    transition_probs = {90 * i: p for i, p in zip(range(4), transition_probs)}

    mdp = MDP(new_grid.tolist(), init_state, terminal_states, discount=discount, transition_probs=transition_probs)
    return mdp


def get_sample_run(mdp, num_sample):
    sampled_transitions = {}
    for (s, a), sp in mdp.transition_table.items():
        if mdp.is_terminal(s):
            possible_states = [mdp.init_state]
            probs = [1]
        else:
            possible_states = list(sp.keys())
            probs = list(sp.values())
        sample = np.random.choice(range(len(possible_states)), num_sample, p=probs)
        sampled_states = [possible_states[i] for i in sample]
        sampled_transitions[(s, a)] = sampled_states

    # Reformat to match data structure expected by rl_provided
    run = {}
    for s in mdp.states:
        run[s] = [sampled_transitions[(s, a)] for a in mdp.actions]

    return run


def save_sample_run(run, world_name):
    with open(f'NEW_world_{world_name}_run.pickle', 'wb') as ff:
        pickle.dump(run, ff)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description="Create Run File")
    parser.add_argument("-world_name", "--world_name", required=True)
    parser.add_argument("-num_samples", "--num_samples", required=True)
    parser.add_argument("-seed", "--seed", required=True)

    args = parser.parse_args()
    world_name = args.world_name
    num_samples = int(args.num_samples)
    seed = int(args.seed)

    np.random.seed(seed)

    mdp = read_secret_world(world_name)
    run = get_sample_run(mdp, num_samples)
    save_sample_run(run, world_name)
