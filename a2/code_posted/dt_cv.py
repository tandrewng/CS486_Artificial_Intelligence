# version 1.1
from typing import List

import dt_global
from dt_core import *


def cv_pre_prune(folds: List, value_list: List[float]) -> (List[float], List[float]):
    """
    Determines the best parameter value for pre-pruning via cross validation.

    Returns two lists: the training accuracy list and the validation accuracy list.

    :param folds: folds for cross validation
    :type folds: List[List[List[Any]]]
    :param value_list: a list of parameter values
    :type value_list: List[float]
    :return: the training accuracy list and the validation accuracy list
    :rtype: List[float], List[float]
    """ 
    t_accuracy = []
    v_accuracy = []
    for v in value_list:
        ti_accuracy = 0
        vi_accuracy = 0
        n = len(folds)
        for i in range(n):
            validation = folds[i]
            training = []
            for j in range(len(folds)):
                if j != i:
                    for k in folds[j]:
                        training.append(k)
            i_dt = learn_dt(training, dt_global.feature_names[:-1], v)
            ti_accuracy += get_prediction_accuracy(i_dt, training, max_depth=v)
            vi_accuracy += get_prediction_accuracy(i_dt, validation, max_depth=v)
        ti_accuracy /= n
        vi_accuracy /= n
        t_accuracy.append(ti_accuracy)
        v_accuracy.append(vi_accuracy)
    return t_accuracy, v_accuracy

def cv_post_prune(folds: List, value_list: List[float]) -> (List[float], List[float]):
    """
    Determines the best parameter value for post-pruning via cross validation.

    Returns two lists: the training accuracy list and the validation accuracy list.

    :param folds: folds for cross validation
    :type folds: List[List[List[Any]]]
    :param value_list: a list of parameter values
    :type value_list: List[float]
    :return: the training accuracy list and the validation accuracy list
    :rtype: List[float], List[float]
    """ 
    t_accuracy = []
    v_accuracy = []
    for v in value_list:
        ti_accuracy = 0
        vi_accuracy = 0
        n = len(folds)
        for i in range(n):
            validation = folds[i]
            training = []
            for j in range(len(folds)):
                if j != i:
                    for k in folds[j]:
                        training.append(k)
            i_dt = learn_dt(training, dt_global.feature_names[:-1])
            post_prune(i_dt, v)
            ti_accuracy += get_prediction_accuracy(i_dt, training, min_num_examples=v)
            vi_accuracy += get_prediction_accuracy(i_dt, validation, min_num_examples=v)
        ti_accuracy /= n
        vi_accuracy /= n
        t_accuracy.append(ti_accuracy)
        v_accuracy.append(vi_accuracy)
    return t_accuracy, v_accuracy
