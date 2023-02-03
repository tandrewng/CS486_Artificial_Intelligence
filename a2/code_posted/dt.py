from dt_provided import *
from dt_core import *
from dt_global import *
from dt_cv import *
from anytree import Node, RenderTree
from anytree.dotexport import RenderTreeGraph
import time

data = read_data('data.csv')

# splits = get_splits(data, 'pox')
# print(splits)

# sp, ig = choose_split(data, 'pox')
# print("info gain: " + str(ig))
# print("optimal split point: " + str(sp))

# feat, sp = choose_feature_split(data, ['pox', 'mcg'])
# print("opt feature: " + feat)
# print("optimal split point: " + str(sp))

# root = learn_dt(data, dt_global.feature_names[:-1])
# print(root.height)
# print(len(root.descendants))
# training_accuracy, validation_accuracy = cv_post_prune([data[1:3], data[3:5]], [1, 2, 3, 4])
# print(training_accuracy, validation_accuracy)
# post_prune(root, 4)
# print(str(RenderTree(root)).encode('utf-8'))
# RenderTreeGraph(root).to_picture("tree3.png")
# print(get_prediction_accuracy(root, data))
# tstart = time.time()
# folds = preprocess(data, 10)
# training_set, validation_set = cv_pre_prune(folds, list(range(0, 31)))
# tstop = time.time()
# print(tstop-tstart)
# training_set, validation_set = cv_post_prune(folds, list(range(0, 301, 20)))
# print(training_set)
# print(validation_set)
# max_val = max(validation_set)
# print(validation_set.index(max_val))