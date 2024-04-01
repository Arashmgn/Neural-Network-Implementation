import numpy as np

def confusion_matrix(y_true, y_pred):
    """
    Compute the confusion matrix.
    Args:
        y_true (numpy.ndarray): True labels (ground truth).
        y_pred (numpy.ndarray): Predicted labels.
    Returns:
        numpy.ndarray: Confusion matrix.
    """
    assert len(y_true) == len(y_pred), "Input arrays must have the same length"
    unique_labels = np.unique(y_true)
    num_classes = len(unique_labels)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(num_classes):
        true_mask = (y_true == unique_labels[i])
        for j in range(num_classes):
            pred_mask = (y_pred == unique_labels[j])
            conf_matrix[i, j] = np.sum(true_mask & pred_mask)

    return conf_matrix

def precision(conf_matrix):
    """
    Compute precision.
    Args:
        conf_matrix (numpy.ndarray): Confusion matrix.
    Returns:
        float: Precision.
    """
    tp = conf_matrix[1, 1]
    fp = conf_matrix[0, 1]
    precision = tp / (tp + fp)
    return precision

def recall(conf_matrix):
    """
    Compute recall.
    Args:
        conf_matrix (numpy.ndarray): Confusion matrix.
    Returns:
        float: Recall.
    """
    tp = conf_matrix[1, 1]
    fn = conf_matrix[1, 0]
    recall = tp / (tp + fn)
    return recall

def f1_score(conf_matrix):
    """
    Compute F1-score.
    Args:
        conf_matrix (numpy.ndarray): Confusion matrix.
    Returns:
        float: F1-score.
    """
    prec = precision(conf_matrix)
    rec = recall(conf_matrix)
    f1 = 2 * (prec * rec) / (prec + rec)
    return f1


def mse_loss(truth, pred):
    """
    Compute the confusion matrix.
    Args:
        truth (numpy.ndarray): True labels (ground truth).
        pred (numpy.ndarray): Prediction.
    Returns:
        numpy.ndarray: Confusion matrix.
    """
    squared_diff = (truth - pred) ** 2
    overall_mse = squared_diff.mean()
    return overall_mse
