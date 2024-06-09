import numpy as np
from scipy.signal import find_peaks
from itertools import product
from typing import Dict

def signal_normalize(signal):
    signal_normalized = normalize_subtract_mean(signal)
    return signal_normalized

def dict_configs(param_dict: Dict, increase_params_simultaneously_flag:bool = False, include_params_border_values_flag:bool=False):
    """
    Function creates parameter grid from the dictionary param_dict
    :param param_dict: dictionary with a structure:
        {
        "param_1_name ":[param_1 values],
        "param_2_name ":[param_2 values],
        ...
        "param_N_name ":[param_N values]
        }
    :param increase_params_simultaneously_flag: flag, indicating the method of grid creation:
        False: All possible combinations of parameters are considered
        True: M combinations of parameters are created. M values of each parameter are provided.
              In each combination of parameters j, j in [0,M-1], j-th value of each parameter is used.
    :return: param_dicts: lists of dictionaries, containing all possible combinations of parameters

    """
    param_dicts = []
    param_array = np.array(list(param_dict.values())).T
    #print(param_array)
    if increase_params_simultaneously_flag:
        for row in param_array[1:]:
            param_dicts.append(dict(zip(param_dict.keys(), row)))

    #additional part:
    # parameters combinations with constant min values of parameter 0 and 2 (window size, do not influence on the magnitude of warping and bamping)
    # and varying values of parameters 1 (time warping rate) and 3 (bump height)
        if include_params_border_values_flag:
            min_param_values = np.min(param_array, axis=0)

            #time warping rate
            warp_rate_param_array = np.zeros_like(param_array)
            warp_rate_param_array[:, :] = min_param_values[:]
            warp_rate_param_array[:, 1] = param_array[:, 1]

            for row in warp_rate_param_array[1:]:
                param_dicts.append(dict(zip(param_dict.keys(), row)))

            # bump height
            bump_height_param_array = np.zeros_like(param_array)
            bump_height_param_array[:, :] = min_param_values[:]
            bump_height_param_array[:, 3] = param_array[:, 3]

            for row in bump_height_param_array[1:]:
                param_dicts.append(dict(zip(param_dict.keys(), row)))

    else:
        for vcomb in product(*param_dict.values()):
            param_dicts.append(dict(zip(param_dict.keys(), vcomb)))

    return param_dicts


def normalize_subtract_mean(var_arr, _min=0, _max=1):# theoretical_num_frames=660):
    """
    Function that normalize and subtract the mean of a provided variable
    :param var_arr: provided variable
    :param _min: min value for the normalization
    :param _max: max value for the normalization
    :param theoretical_num_frames: number of frames theoretically present in the sequence

    return: Normalized and mean subtracted variable
    """
    var_arr = max_min_scaler(var_arr,
                             new_min=_min,
                             new_max=_max,
                             percentile_scaling=False)

    var_arr = var_arr - np.mean(var_arr)
    return var_arr

def max_min_scaler(data, new_min=0, new_max=1, percentile_scaling=False, min_percentile=15, max_percentile=85):
    """
    :param data: input data as array
    :param new_min: new minimum value for the normalization range
    :param new_max: new maximum value for the normalization range
    :param percentile_scaling: set True if a scaling in a percentile range is wanted (default True)
    :param min_percentile: percentile value below which the data is clipped
    :param max_percentile: percentile value above which the data is clipped

    compute statistic over data to avoid taking into consideration the values out of the real features range

    :return: normalized range within the defined range
    """
    data = np.asarray(data)

    if percentile_scaling:
        percentiles = np.percentile(data, [min_percentile, max_percentile])

        data = np.clip(data, percentiles[0], percentiles[1])

    max_val = np.amax(data)
    min_val = np.amin(data)

    normalization_coefficient = (new_max - new_min) / (max_val - min_val)
    normalized_data = normalization_coefficient * (data - max_val) + new_max
    normalized_data[np.isnan(normalized_data)] = 0.000001

    return normalized_data


def find_signal_peaks(signal, peaks_distance = 12, peaks_height = 0.03, prominence = 0.04):

    peaks = find_peaks(x=signal,
                       height=peaks_height,
                       distance=peaks_distance,
                       prominence=prominence)[0]
    peaks = remove_redundant_peaks(peaks, len(signal))
    return peaks

def remove_redundant_peaks(signal_peaks, signal_len):
    indices_to_remove = [0, 1, signal_len - 1, signal_len - 2]
    signal_peaks_upd = []

    for i, idx in enumerate(signal_peaks):
        if i > 0 and abs(idx - signal_peaks[i - 1]) < 2:
            continue  # Skip this index
        if idx not in indices_to_remove:
            signal_peaks_upd.append(idx)
    return signal_peaks_upd

def find_matching_peak_pairs(radar_peaks,reference_peaks, peak_distance_matrix):
    rows, cols = peak_distance_matrix.shape
    mask = np.zeros_like(peak_distance_matrix)

    # Find minimum element along each row
    for i in range(rows):
        min_index = np.argmin(peak_distance_matrix[i,:])
        mask[i, min_index] += 1


    # Find minimum element along each column

    for j in range(cols):
        min_index = np.argmin(peak_distance_matrix[:, j])

        mask[min_index, j] += 1


    # Extract pairs of indices where the value in the mask is 2

    pairs = []
    previous_min_idx_column = 0
    for i in range(rows):
        check_pair_element = np.where(mask[i,previous_min_idx_column:] == 2)[0]
        if check_pair_element.size>0:
            j=check_pair_element[0]
            j+=previous_min_idx_column
            previous_min_idx_column = j+1
            pairs.append([radar_peaks[i],reference_peaks[j]])

    return pairs

