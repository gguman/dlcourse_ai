import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    positive = np.nonzero(prediction)
    negative = np.nonzero(prediction == 0)
    
    true_positive = np.sum(prediction[positive] == ground_truth[positive])
    false_positive = np.sum(prediction[positive] != ground_truth[positive])
    
    true_negative = np.sum(prediction[negative] == ground_truth[negative])
    false_negative = np.sum(prediction[negative] != ground_truth[negative])
    
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    
    f1 = 2 / (pow(precision, -1) + pow(recall, -1))
    
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    #import ipdb; ipdb.set_trace()
    
    # TODO: Implement computing accuracy
    #return 0
    
    accuracy = np.sum(prediction == ground_truth)/len(prediction)
    
    return accuracy