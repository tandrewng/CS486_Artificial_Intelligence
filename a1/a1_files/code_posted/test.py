from board import *
from pprint import pprint
from solve import *
import time
boards = from_file("jams_posted.txt")

# pprint(vars(boards[0]))


# ['<', '>', '.', '.', '.', '^'],
# ['^', '.', '.', '^', '.', '|'],
# ['|', '<', '>', '|', '.', 'v'],
# ['v', '.', '.', 'v', '.', '.'],
# ['^', '.', '.', '.', '<', '>'],
# ['v', '.', '<', '-', '>', '.']
for board in boards:

    #start = time.time()
    path, cost, loop = a_star(board,advanced_heuristic)
    #end = time.time()

    print (loop)
    # for s in path:
    #     s.board.display()
    #     print(s.f)
    #     print("hash", s.id)

# print("time taken:", end - start)

# successors = get_successors(state)

# for s in successors:
#     s.board.display()
#     print(s.f)