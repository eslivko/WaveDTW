import numpy as np

from tslearn.metrics import dtw_path_from_metric
def classic_dtw_distance(signal, reference_signal, x_signal, x_ref_signal,  distance_type = "euclidean", cost_normalized = True, dDTW=False):
    """
    Implementation based on pyts.metrics.dtw
    :param signal:
    :param reference_signal:
    :param x_signal:
    :param x_ref_signal:
    :param distance_type:
    :param cost_normalized:
    :param dDTW:
    :return:
    """
    n = len(signal)
    m = len(reference_signal)

    path, total_cost = dtw_path_from_metric(signal, reference_signal, metric = distance_type)
    cost_matrix = np.full((n, m), -1000) #TODO:make a parameter for cost matrix calculation and implement calculation separately
    if cost_normalized:
        if n >= m:
            normalization_len = n
        else:
            normalization_len = m
        total_cost = total_cost / normalization_len
    return total_cost, path,cost_matrix
