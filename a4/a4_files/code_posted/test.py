import pickle

objects = []
output = open("output.txt", "w")
with (open("world_lecture_run.pickle", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
print(objects, file=output)
output.close()