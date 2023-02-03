from typing import List

class Car:
    """
    Represents one car on the puzzle board.
    """

    def __init__(self, coord_x: int, coord_y: int, orientation: str, length: int, is_goal: bool):
        """
        Stores the car's location, orientation, length, 
        and whether it's the goal car or not.

        The location of each car is the (x, y) coordinates of its 
        upper left corner. The x coordinate is its column position. 
        The y coordinate is its row position.

        A car can move either horizontally or vertically.  Therefore, 
        for each car's location, one coordinate is fixed (fix_coord) and 
        the other coordinate (var_coord) can change when the car moves.
        For example, for a horizontal car, the fixed coordinate is y and
        the variable coordinate is x.

        :param coord_x: The x coordinate of the car.
        :type coord_x: int
        :param coord_y: The y coordinate of the car
        :type coord_y: int
        :param orientation: The orientation of the car (one of 'h' or 'v')
        :type orientation: str
        :param length: The length of the car.
        :type length: int
        :param is_goal: True if the car is the goal car and False otherwise.
        :type is_goal: bool
        """

        # var_coord stores the value of the coordinate that CAN vary.
        self.var_coord = coord_x if orientation == 'h' else coord_y

        # fix_coord stores the value of the coordinate that CANNOT vary.
        self.fix_coord = coord_x if orientation == 'v' else coord_y

        self.orientation = orientation
        self.length = length
        self.is_goal = is_goal

    def set_coord(self, coord):
        """
        Update the var_coord of the car after moving.

        :param coord: The new coordination after moving.
        :type coord: int
        """

        self.var_coord = coord


class Board:
    """
    Represents the puzzle board.

    """

    def __init__(self, name: str, size: int, cars: List[Car]):
        """
        Stores the board's name, board's size and the list of cars on the board.

        :param name: The name of the board.
        :type name: str
        :param size: The side length of the square board.
        :type size: int
        :param cars: The list of Cars
        :type cars: List[Car]
        """

        self.name = name
        self.size = size
        self.cars = cars

        # When creating the board, the constructor automatically 
        # creates a grid. The grid is a 2-d (size * size) array
        # containing symbols to represent the content of the board.
        # The grid is used to display the board.
        # You may also want to use the grid to determine
        # whether a move is legal or not.
        self.grid = []
        self.__exit_info = None
        self.__construct_grid()

    # customized eq for object comparison.
    def __eq__(self, other):
        if isinstance(other, Board):
            return self.grid == other.grid
        return False

    # customized hash for hashing an object.
    def __hash__(self):
        return hash(tuple(map(tuple, self.grid)))

    def __construct_grid(self):
        """
        Called in __init__ to set up a 2-d grid based on car location information.

        """

        for i in range(self.size):
            line = []
            for j in range(self.size):
                line.append('.')
            self.grid.append(line)
        for car in self.cars:
            if car.orientation == 'h':
                self.grid[car.fix_coord][car.var_coord] = '<'
                for i in range(car.length - 2):
                    self.grid[car.fix_coord][car.var_coord + i + 1] = '-'
                self.grid[car.fix_coord][car.var_coord + car.length - 1] = '>'
            elif car.orientation == 'v':
                self.grid[car.var_coord][car.fix_coord] = '^'
                for i in range(car.length - 2):
                    self.grid[car.var_coord + i + 1][car.fix_coord] = '|'
                self.grid[car.var_coord + car.length - 1][car.fix_coord] = 'v'
            if car.is_goal:
                self.__exit_info = (car.fix_coord, car.orientation)

    def display(self):
        """
        Prints out the board in a human readable format.
        """

        def print_cap(exit_col=None):
            print('+', end='')
            for i in range(self.size * 2 + 1):
                if int(i / 2) == exit_col:
                    print(' ', end='')
                else:
                    print('-', end='')
            print('+')

        def print_grid_line(line, exit_line=None):
            print('|', end='')
            for i in range(self.size * 2):
                if i % 2 == 0:
                    print(' ', end='')
                else:
                    print(line[int(i / 2)], end='')
            print(' ', end='')
            if exit_line is None:
                print('|')
            else:
                print(' ')

        print_cap()
        for i, line in enumerate(self.grid):
            if self.__exit_info[1] == 'h' and i == self.__exit_info[0]:
                print_grid_line(line, self.__exit_info[0])
            else:
                print_grid_line(line)
        print_cap(self.__exit_info[0] if self.__exit_info[1] == 'v' else None)


class State:
    """
    State class wrapping a Board with some extra current state information.
    Note that State and Board are different. Board has the locations of the cars. 
    State has a Board and some extra information that is relevant to the search: 
    heuristic function, f value, current depth and parent.
    """

    def __init__(self, board: Board, hfn, f: int, depth: int, parent=None):
        """
        :param board: The board of the state.
        :type board: Board
        :param hfn: The heuristic function.
        :type hfn: Optional[Heuristic]
        :param f: The f value of current state.
        :type f: int
        :param depth: The depth of current state in the search tree. Depth of the root node is 0.
        :type depth: int
        :param parent: The parent of current state.
        :type parent: Optional[State]
        """
        self.board = board
        self.hfn = hfn
        self.f = f
        self.depth = depth
        self.parent = parent
        self.id = hash(board)  # The id for breaking ties.

    # customized eq for object comparison.
    def __eq__(self, other):
        if isinstance(other, State):
            return self.f == other.f and self.id == other.id
        return False

    # customized lt for object comparison.
    def __lt__(self, other):
        return self.f < other.f


def zero_heuristic(board: Board):
    """
    Simply return zero for any state.
    """

    return 0


def from_file(filename: str) -> List[Board]:
    """
    Reads in all the puzzles in the given file 
    and returns a list of the initial boards 
    for all the puzzles. 
    
    :param filename: The name of the given file.
    :type filename: str
    :return: A list of loaded Boards.
    :rtype: List[Board]
    """

    puzzle_file = open(filename, "r")
    counter = 0
    board_size = 0
    board_name = ""
    cars = []
    boards = []
    for line in puzzle_file:
        if line.split()[0] == ".":
            boards.append(Board(board_name, board_size, cars))
            counter = 0
            cars = []
            continue
        if counter == 0: # first line has name of puzzle
            board_name = line
        elif counter == 1: # second line has board size
            board_size = int(line)
        else: # the following lines describe cars
            car_info = line.split()
            car_info = [int(x) if i != 2 else x for i, x in enumerate(car_info)]
            car_info.append(True if counter == 2 else False)
            cars.append(Car(*car_info))
        counter += 1
    puzzle_file.close()
    return boards
