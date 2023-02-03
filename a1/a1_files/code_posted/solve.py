from os import stat
from board import *
import copy

def a_star(init_board, hfn):
    """
    Run the A_star search algorithm given an initial board and a heuristic function.

    If the function finds a goal state, it returns a list of states representing
    the path from the initial state to the goal state in order and the cost of
    the solution found.
    Otherwise, it returns am empty list and -1.

    :param init_board: The initial starting board.
    :type init_board: Board
    :param hfn: The heuristic function.
    :type hfn: Heuristic
    :return: (the path to goal state, solution cost)
    :rtype: List[State], int
    """

    init_state = State(init_board, hfn, 0, 0)
    init_list = [init_state]
    frontier = {0: init_list}
    explored = set()
    loops = 0
    while frontier:
        
        minCost = min(frontier.keys())
        states = frontier[minCost]
        currNode = states[0]
        currNodeIndex = 0
        for i in range(1, len(states)):
            if states[i] == currNode:
                if states[i].parent.id < currNode.parent.id:
                    currNode = states[i]
                    currNodeIndex = i
            elif (states[i].f == currNode.f) and (states[i].id < currNode.id):
                currNode = states[i]
                currNodeIndex = i
        currNode = states.pop(currNodeIndex)
        if not frontier[minCost]:
            frontier.pop(minCost)

        if not currNode.board in explored:
            explored.add(currNode.board)
            if (is_goal(currNode)):
                return get_path(currNode), currNode.depth, loops
            successors = get_successors(currNode)
            loops += 1
            for successor in successors:
                if not successor.f in frontier:
                    frontier[successor.f] = [successor]
                else:
                    frontier[successor.f].append(successor)
    return [], -1, loops


def dfs(init_board):
    """
    Run the DFS algorithm given an initial board.

    If the function finds a goal state, it returns a list of states representing
    the path from the initial state to the goal state in order and the cost of
    the solution found.
    Otherwise, it returns am empty list and -1.

    :param init_board: The initial board.
    :type init_board: Board
    :return: (the path to goal state, solution cost)
    :rtype: List[State], int
    """
    init_state = State(init_board, zero_heuristic, 0, 0)
    frontier = [init_state]
    explored = set()
    while frontier:
        currNode = frontier.pop()
        if not currNode.board in explored:
            explored.add(currNode.board)
            if (is_goal(currNode)):
                return get_path(currNode), currNode.f
            successors = get_successors(currNode)
            while successors:
                maxIdState = successors[0]
                maxIdIndex = 0
                for i in range(1, len(successors)):
                    if maxIdState.id < successors[i].id:
                        maxIdState = successors[i]
                        maxIdIndex = i
                frontier.append(successors.pop(maxIdIndex))

    return [], -1


def moveBack(grid, car):
    newCars = []
    i = car.var_coord
    if i == 0:
        return newCars
    while (i > 0):
        if (car.orientation == 'h'):
            if grid[car.fix_coord][i-1] == '.':
                newCar = copy.deepcopy(car)
                newCar.var_coord = i - 1
                newCars.append(newCar)
            else:
                break
        if (car.orientation == 'v'):
            if grid[i-1][car.fix_coord] == '.':
                newCar = copy.deepcopy(car)
                newCar.var_coord = i - 1
                newCars.append(newCar)
            else:
                break
        i -= 1
    return newCars

def moveForward(grid, car):
    newCars = []
    i = car.var_coord
    if i == (6-car.length):
        return newCars
    while (i < (6-car.length)):
        if (car.orientation == 'h'):
            if grid[car.fix_coord][i+car.length] == '.':
                newCar = copy.deepcopy(car)
                newCar.var_coord = i + 1
                newCars.append(newCar)
            else:
                break
        if (car.orientation == 'v'):
            if grid[i+car.length][car.fix_coord] == '.':
                newCar = copy.deepcopy(car)
                newCar.var_coord = i + 1
                newCars.append(newCar)
            else:
                break
        i += 1
    return newCars

def get_successors(state):
    """
    Return a list containing the successor states of the given state.
    The states in the list may be in any arbitrary order.

    :param state: The current state.
    :type state: State
    :return: The list of successor states.
    :rtype: List[State]
    """

    newStates = []
    for i in range(len(state.board.cars)):
        currCar = state.board.cars[i]
        newCars = []
        newCars = newCars + moveBack(state.board.grid, currCar)
        newCars = newCars + moveForward(state.board.grid, currCar)

        # create new states using new valic cars
        for newCar in newCars:
            newBoardCars = copy.deepcopy(state.board.cars)
            newBoardCars[i] = newCar
            newBoard = Board(state.board.name, state.board.size, newBoardCars)
            newF = state.depth + 1 + state.hfn(newBoard)
            newState = State(newBoard, state.hfn, newF, state.depth + 1, state)
            newStates.append(newState)
            
    return newStates


def is_goal(state):
    """
    Returns True if the state is the goal state and False otherwise.

    :param state: the current state.
    :type state: State
    :return: True or False
    :rtype: bool
    """

    return (state.board.cars[0].var_coord == 4)

def get_path(state):
    """
    Return a list of states containing the nodes on the path 
    from the initial state to the given state in order.

    :param state: The current state.
    :type state: State
    :return: The path.
    :rtype: List[State]
    """
    spath = []
    currState = state

    while not currState == None:
        spath.insert(0, currState)
        currState = currState.parent

    return spath


def blocking_heuristic(board):
    """
    Returns the heuristic value for the given board
    based on the Blocking Heuristic function.

    Blocking heuristic returns zero at any goal board,
    and returns one plus the number of cars directly
    blocking the goal car in all other states.

    :param board: The current board.
    :type board: Board
    :return: The heuristic value.
    :rtype: int
    """

    
    goalCar = board.cars[0]
    if goalCar.var_coord == 4 :
        return 0
    checkRow = board.grid[2]
    blocking = 1
    for i in range(goalCar.var_coord + 2, len(checkRow)):
        if checkRow[i] != '.':
            blocking += 1
    return blocking

def advanced_heuristic(board):
    """
    An advanced heuristic of your own choosing and invention.

    :param board: The current board.
    :type board: Board
    :return: The heuristic value.
    :rtype: int
    """
    goalCar = board.cars[0]
    if goalCar.var_coord == 4 :
        return 0
    
    blockingCarsi = []
    for i in range(1, len(board.cars)):
        if board.cars[i].orientation == 'h':
            if (board.cars[i].fix_coord == 2) and \
                (board.cars[i].var_coord > goalCar.var_coord):
                blockingCarsi.append(board.cars[i])
        if board.cars[i].orientation == 'v':
            if (board.cars[i].fix_coord >= (goalCar.var_coord + goalCar.length)) and \
            (board.cars[i].var_coord <= goalCar.fix_coord and \
            (board.cars[i].var_coord + board.cars[i].length) >= goalCar.fix_coord):
                blockingCarsi.append(board.cars[i])

    blocking = 1 + len(blockingCarsi)
    for car in blockingCarsi:
        if car.orientation == 'h':
            if (car.var_coord == 0 or board.grid[car.fix_coord][car.var_coord-1] != '.') and \
            (car.var_coord + car.length - 1 == 5 or board.grid[car.fix_coord][car.var_coord + car.length] != '.'):
                blocking += 1
        if car.orientation == 'v':
            if (car.var_coord == 0 or board.grid[car.fix_coord][car.var_coord-1] != '.') and \
            (car.var_coord + car.length - 1 == 5 or board.grid[car.var_coord + car.length][car.fix_coord] != '.'):
                blocking += 1

    return blocking
    