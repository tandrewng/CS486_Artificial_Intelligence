# version 1.1
import math
from typing import List
from anytree import Node, RenderTree
import numpy as np
from collections import Counter

import dt_global 

count = 0

def get_splits(examples: List, feature: str) -> List[float]:
    """
    Given some examples and a feature,
    returns a list of potential split point values for the feature.
    any
    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param feature: a feature
    :type feature: str
    :return: a list of potential split point values 
    :rtype: List[float]
    """ 
    splits = []
    f_index = dt_global.feature_names.index(feature)
    sorted_examples = sorted(examples, key=lambda x: x[f_index])
    n = len(sorted_examples)
    vals_labels = {}
    i = 0
    past_val = sorted_examples[i][f_index]
    while i < n:
        i_labels = set()
        i_val = sorted_examples[i][f_index]
        new_i = i
        i_add = sorted_examples[new_i][f_index]
        while i_add == i_val and new_i < n:
            i_labels.add(sorted_examples[new_i][dt_global.label_index])
            new_i += 1
            if (new_i >= n): break
            i_add = sorted_examples[new_i][f_index]
        i = new_i
        
        vals_labels[i_val] = i_labels

        if (vals_labels[past_val] != vals_labels[i_val]):
            splits.append((past_val + i_val)/2)
        
        past_val = i_val

    return splits

def entropy(examples: List):
    n = len(examples)
    entropy = 0
    x_classes = []
    for x in examples:
        x_classes.append(x[dt_global.label_index])
    
    x_classes_count = Counter(x_classes)
    for c in x_classes_count:
        p = x_classes_count[c]/n
        entropy += -p * math.log2(p)
    return entropy

def split_examples(examples: List, feature: str, split: float) -> (List, List):
    """
    Given some examples, a feature, and a split point,
    splits examples into two lists and return the two lists of examples.

    The first list of examples have their feature value <= split point.
    The second list of examples have their feature value > split point.

    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param feature: a feature
    :type feature: str
    :param split: the split point
    :type split: float
    :return: two lists of examples split by the feature split
    :rtype: List[List[Any]], List[List[Any]]
    """ 
    left = list()
    right = list()
    f_index = dt_global.feature_names.index(feature)
    for x in examples:
        if x[f_index] <= split:
            left.append(x)
        else:
            right.append(x)
    return left, right

def choose_split(examples: List, feature: str):
    examples_entropy = entropy(examples)
    best_info_gain = -1
    best_split_val = -1
    splits = get_splits(examples, feature)
    n = len(examples)
    for split in splits:
        left, right = split_examples(examples, feature, split)
        w_left = len(left) / n * entropy(left)
        w_right = len(right) / n * entropy(right)
        info_gain = examples_entropy - (w_left + w_right)
        if ((info_gain > best_info_gain) or (math.isclose(info_gain, best_info_gain, abs_tol = 1e-5) and split > best_split_val)):
            best_info_gain = info_gain
            best_split_val = split
    return best_split_val, best_info_gain

def choose_feature_split(examples: List, features: List[str]) -> (str, float):
    """
    Given some examples and some features,
    returns a feature and a split point value with the max expected information gain.

    If there are no valid split points for the remaining features, return None and -1.

    Tie breaking rules:
    (1) With multiple split points, choose the one with the smallest value. 
    (2) With multiple features with the same info gain, choose the first feature in the list.

    :param examples: a set of examples
    :type examples: List[List[Any]]    
    :param features: a set of features
    :type features: List[str]
    :return: the best feature and the best split value
    :rtype: str, float
    """
    best_feature = None
    best_split_gain = -1
    best_split_val = -1

    for feature in features:
        f_split_val, f_split_gain = choose_split(examples, feature)
        if (best_feature == None) or (f_split_gain > best_split_gain) or (math.isclose(f_split_gain, best_split_gain, abs_tol = 1e-5) and
            (dt_global.feature_names.index(feature) < dt_global.feature_names.index(best_feature))):
            best_split_gain = f_split_gain
            best_split_val = f_split_val
            best_feature = feature
    if (best_split_val == -1): best_feature = None
    return best_feature, best_split_val
        
def decide(examples: List):
    vals = []
    for x in examples:
        vals.append(x[dt_global.label_index])
    return Counter(vals).most_common(1)[0][0]
    
def split_node(cur_node: Node, examples: List, features: List[str], max_depth=math.inf):
    """
    Given a tree with cur_node as the root, some examples, some features, and the max depth,
    grows a tree to classify the examples using the features by using binary splits.

    If cur_node is at max_depth, makes cur_node a leaf node with majority decision and return.

    This function is recursive.

    :param cur_node: current node
    :type cur_node: Node
    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param features: a set of features
    :type features: List[str]
    :param max_depth: the maximum depth of the tree
    :type max_depth: int
    """ 
    global count
    cur_node.decision = decide(examples)
    cur_node.examples = examples

    if (cur_node.depth == max_depth): 
        cur_node._children = 0
        return

    classification_set = set()
    for data in examples:
        classification_set.add(data[-1])
    if len(classification_set) == 1:
        cur_node.decision = classification_set.pop()
        return

    split_feature, split_val = choose_feature_split(examples, features)
    if (split_val == -1):
        cur_node._children = 0
        return

    left, right = split_examples(examples, split_feature, split_val)

    cur_node.feature = split_feature
    cur_node.split = split_val
    if (left and right):
        left_name = str(cur_node.depth + 1) + split_feature + str(split_val) + "L" + str(count)
        left_child = Node(name=left_name, parent=cur_node, feature=None, split=None, examples=None, decision=None)
        count += 1
        split_node(left_child, left, features, max_depth)
        right_name = str(cur_node.depth + 1) + split_feature + str(split_val) + "R" + str(count)
        right_child = Node(name=right_name, parent=cur_node, feature=None, split=None, examples=None, decision=None)
        count += 1
        split_node(right_child, right, features, max_depth)

def learn_dt(examples: List, features: List[str], max_depth=math.inf) -> Node:
    """
    Given some examples, some features, and the max depth,
    creates the root of a decision tree, and
    calls split_node to grow the tree to classify the examples using the features, and
    returns the root node.

    This function is a wrapper for split_node.

    Tie breaking rule:
    If there is a tie for majority voting, always return the label with the smallest value.

    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param features: a set of features
    :type features: List[str]
    :param max_depth: the max depth of the tree
    :type max_depth: int, default math.inf
    :return: the root of the tree
    :rtype: Node
    """ 
    root = Node(name="root")
    split_node(root, examples, features, max_depth)
    return root


def predict(cur_node: Node, example, max_depth=math.inf, \
    min_num_examples=0) -> int:
    """
    Given a tree with cur_node as its root, an example, and optionally a max depth,
    returns a prediction for the example based on the tree.

    If max_depth is provided and we haven't reached a leaf node at the max depth, 
    return the majority decision at this node.

    If min_num_examples is provided and the number of examples at the node is less than
    min_num_examples, return the majority decision at this node.
    
    This function is recursive.

    Tie breaking rule:
    If there is a tie for majority voting, always return the label with the smallest value.

    :param cur_node: cur_node of a decision tree
    :type cur_node: Node
    :param example: one example
    :type example: List[Any]
    :param max_depth: the max depth
    :type max_depth: int, default math.inf
    :param min_num_examples: the minimum number of examples at a node
    :type min_num_examples: int, default 0
    :return: the decision for the given example
    :rtype: int
    """
    
    if (max_depth == cur_node.depth) or (min_num_examples > len(cur_node.examples) or (cur_node.is_leaf)):
        return cur_node.decision
    f_index = dt_global.feature_names.index(cur_node.feature)
    if (example[f_index] <= cur_node.split):
        return predict(cur_node.children[0], example, max_depth, min_num_examples)
    else:
        return predict(cur_node.children[1], example, max_depth, min_num_examples)


def get_prediction_accuracy(cur_node: Node, examples: List, max_depth=math.inf, \
    min_num_examples=0) -> float:
    """
    Given a tree with cur_node as the root, some examples, 
    and optionally the max depth or the min_num_examples, 
    returns the accuracy by predicting the examples using the tree.

    The tree may be pruned by max_depth or min_num_examples.

    :param cur_node: cur_node of the decision tree
    :type cur_node: Node
    :param examples: the set of examples. 
    :type examples: List[List[Any]]
    :param max_depth: the max depth
    :type max_depth: int, default math.inf
    :param min_num_examples: the minimum number of examples at a node
    :type min_num_examples: int, default 0
    :return: the prediction accuracy for the examples based on the cur_node
    :rtype: float
    """ 
    n = len(examples)
    prediction = []

    for x in examples:
        prediction.append(predict(cur_node, x, max_depth, min_num_examples))

    accuracy = 0
    for i in range(n):
        if prediction[i] == examples[i][dt_global.label_index]:
            accuracy += 1
    return accuracy / n


def post_prune(cur_node: Node, min_num_examples: float):
    """
    Given a tree with cur_node as the root, and the minimum number of examples,
    post prunes the tree using the minimum number of examples criterion.

    This function is recursive.

    Let leaf parents denote all the nodes that only have leaf nodes as its descendants. 
    Go through all the leaf parents.
    If the number of examples at a leaf parent is smaller than the pre-defined value,
    convert the leaf parent into a leaf node.
    Repeat until the number of examples at every leaf parent is greater than
    or equal to the pre-defined value of the minimum number of examples.

    :param cur_node: the current node
    :type cur_node: Node
    :param min_num_examples: the minimum number of examples
    :type min_num_examples: float
    """
    if cur_node.is_leaf: return
    if (cur_node.children[0].is_leaf and cur_node.children[1].is_leaf and 
        len(cur_node.examples) < min_num_examples):
        cur_node.children = []
        if (cur_node.parent is not None): post_prune(cur_node.parent, min_num_examples)
    else:
        for child in cur_node.children:
            post_prune(child, min_num_examples)