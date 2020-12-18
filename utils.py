import numpy as np 
import pandas as pd 

def get_confusion_matrix(y_hats, ys, names=None, one_hot=True, plot=True):
    if one_hot:
        y_hats = np.argmax(y_hats, 1)
        ys = np.argmax(ys, 1)
    if names is None:
        names = list(np.unique(ys))
    confusion_matrix = np.zeros((len(names), len(names)))
    for y_hat, y in zip(y_hats, ys):
        confusion_matrix[y_hat, y] += 1
    confusion_matrix = pd.DataFrame(confusion_matrix, columns=list(map(lambda x: 'Actual '+ str(x), names)), index=list(map(lambda x: 'Predicted ' + str(x), names)))
    if plot:
        print(confusion_matrix)
    return confusion_matrix

def class_precisions_recalls(y_hats, ys, one_hot=True, epsilon=1e-8):
    confusion_matrix = get_confusion_matrix(y_hats, ys, one_hot=one_hot, plot=False).values
    precisions = np.diag(confusion_matrix / (np.sum(confusion_matrix, 1) + epsilon)) * 100.0
    recalls = np.diag(confusion_matrix / (np.sum(confusion_matrix, 0) + epsilon)) * 100.0
    return precisions, recalls


def accuracy(y_hats, ys, one_hot=True):
    if one_hot:
        y_hats = np.argmax(y_hats, 1)
        ys = np.argmax(ys, 1)
    return np.sum(y_hats == ys) / len(ys) * 100.0

def _get_true_positives(y_hats, ys):
    true_positives = np.zeros(len(np.unique(ys)))
    for y_hat, y in zip(y_hats, ys):
        if y_hat == y:
            true_positives[y] += 1
    return true_positives

def _get_false_positives(y_hats, ys):
    false_positives = np.zeros(len(np.unique(ys)))
    for y_hat, y in zip(y_hats, ys):
        if y_hat != y:
            false_positives[y] += 1
    return false_positives

def _get_false_negatives(y_hats, ys):
    false_negatives = np.zeros(len(np.unique(ys)))
    for y_hat, y in zip(y_hats, ys):
        if y_hat != y:
            false_negatives[y_hat] += 1
    return false_negatives

def _get_true_positives_micro(y_hats, ys):
    true_positives = np.zeros(ys.shape[1])
    for y_hat, y in zip(y_hats, ys):
        true_positives += (y_hat == y) & (y == 1.)
    return true_positives

def _get_false_positives_micro(y_hats, ys):
    false_positives = np.zeros(ys.shape[1])
    for y_hat, y in zip(y_hats, ys):
        false_positives += (y_hat != y) & (y == 0.)
    return false_positives

def _get_false_negatives_micro(y_hats, ys):
    false_negatives = np.zeros(ys.shape[1])
    for y_hat, y in zip(y_hats, ys):
        false_negatives += (y_hat != y) & (y == 1.)
    return false_negatives

def f1_score_multi_class(y_hats, ys, one_hot=True, average='micro', epsilon=1e-12):
    if average == 'micro':
        if len(y_hats.shape) == 1:
            y_hats = pd.get_dummies(y_hats).values
            ys = pd.get_dummies(ys).values
        true_positives = _get_true_positives_micro(y_hats, ys)
        false_positives = _get_false_positives_micro(y_hats, ys)
        false_negatives = _get_false_negatives_micro(y_hats, ys)
        precision = np.sum(true_positives) / np.sum(false_positives + true_positives + epsilon)
        recall = np.sum(true_positives) / np.sum(false_negatives + true_positives + epsilon)
        return 2 * precision * recall / (precision + recall + epsilon)
    else:
        if one_hot:
            y_hats = np.argmax(y_hats, 1)
            ys = np.argmax(ys, 1)
        true_positives = _get_true_positives(y_hats, ys)
        false_positives = _get_false_positives(y_hats, ys)
        false_negatives = _get_false_negatives(y_hats, ys)
        precisions = true_positives / (false_positives + true_positives + epsilon)
        recalls = true_positives / (false_negatives + true_positives + epsilon)
        return np.sum(2 * precisions * recalls / (precisions + recalls + epsilon)) / len(true_positives)