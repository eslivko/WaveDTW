import numpy as np

from ClassicDTW import *
from functions import *
from scipy.signal import hilbert



def get_derivative(signal):
    signal_padded = np.pad(signal, (1, 0), constant_values=(signal[0]))
    signal_padded = np.pad(signal_padded, (0, 1), constant_values=(signal[-1]))
    signal_derivative = ((signal_padded[1:len(signal_padded) - 1] - signal_padded[0:len(signal_padded) - 2]) + (
            signal_padded[2:len(signal_padded)] - signal_padded[0:len(signal_padded) - 2]) / 2) / 2
    return signal_derivative

def hilbert_transform(signal):
    """

    :param signal:
    :return:
    """
    analytic_signal = hilbert(signal)

    amplitude_envelope = np.abs(analytic_signal)
    amplitude_envelope = get_derivative(amplitude_envelope)

    instantaneous_phase = np.unwrap(np.angle(analytic_signal))  # * 180 / np.pi
    instantaneous_phase = get_derivative(instantaneous_phase)


    return amplitude_envelope, instantaneous_phase

def hilbert_transform_features(signal):
    amplitude_envelope_norm, instantaneous_phase_norm = hilbert_transform(signal)
    signal_transformed = np.column_stack((amplitude_envelope_norm, instantaneous_phase_norm))

    return signal_transformed

def dtw_distance_wave(signal,
                      reference_signal,
                      signal_peaks,
                      reference_peaks,
                      x_signal,
                      x_ref_signal,
                      distance_type = "euclidean",
                      peak_matching_algorithm = "min_distance",
                      cost_normalized = True,
                      use_segmentation=True,
                      dDTW=False):
    """
    Function calculate DTW distance for 2 wave signals

    :param: signal: 1d ndarray with signal values,
    :param: reference_signal: 1d ndarray with reference signal values,
    :param: radar_peaks: list of indicies of peaks in radar signal,
    :param: reference_peaks: list of indicies of peaks in reference signal
    :param: x_seconds: time x-axis
    :param: distance_type: possible values:
            "euclidean",
            "absolute"
    :param: cost_normalized: bool parameter for cost normalization
    :param: peak_matching_algorithm: type of algorithm for peaks matching, possible values:
            "min_distance" - approach based on finding peak pairs with mutual minimum distance,
            "opt" - optimization based approach
            "none" - no peak matching is applied. triggers general DTW implementation to work

    :return cost_accumulative, path_accumulative: total distance and path across which the distance was calculated
    """


    if not use_segmentation:

        signal = hilbert_transform_features(signal)
        reference_signal = hilbert_transform_features(reference_signal)

        cost_accumulative, path_accumulative, total_cost_matrix = classic_dtw_distance(
            signal=signal,
            reference_signal=reference_signal,
            x_signal=x_signal,
            x_ref_signal=x_ref_signal,
            distance_type=distance_type,
            cost_normalized=cost_normalized,
            dDTW=dDTW)
        path_node_pairs = None


    else:


        peak_distance_matrix = np.full((len(signal_peaks), len(reference_peaks)), np.inf)
        for i in range(len(signal_peaks)):
            for j in range(len(reference_peaks)):
                #criteria for peak matching - alignment in time
                peak_distance_matrix[i,j] = abs(signal_peaks[i] - reference_peaks[j])


        ##Find the node pairs and add pairs of first and last elements of signal and reference
        node_pairs = []
        if (len(signal_peaks)>0 and len(reference_peaks)>0):

            ### Algorithm 2 (Naive):
            if peak_matching_algorithm=="min_distance":
                node_pairs = find_matching_peak_pairs(signal_peaks,reference_peaks, peak_distance_matrix)
        node_pairs.insert(0, [0, 0])
        node_pairs.append([len(signal) - 1, len(reference_signal) - 1])

        #dictionary total cost include info by the segments: [distance: float, path:List]
        #total_cost[(idx_point_radar,idx_point_refefence)-(idx_point_radar,idx_point_refefence)] = distance, [path]

        total_cost={}
        total_cost_matrix = np.full((len(signal), len(reference_signal)), np.inf)
        total_cost_matrix[0,0] = 0
        #estimate distance between nodes (peaks)
        for i,node in enumerate(node_pairs):
            if i == 0:
                continue
            node_radar_prev, node_reference_prev = node_pairs[i-1]
            node_radar, node_reference = node

            signal_segment = signal[node_radar_prev:node_radar + 1]
            x_signal_segment = x_signal[node_radar_prev:node_radar + 1]

            ref_signal_segment = reference_signal[node_reference_prev:node_reference + 1]
            x_ref_signal_segment = x_ref_signal[node_reference_prev:node_reference + 1]

            signal_segment = hilbert_transform_features(signal_segment)
            ref_signal_segment = hilbert_transform_features(ref_signal_segment)


            segment_distance, segment_path, segment_cost_matrix = classic_dtw_distance(
                                                                        signal = signal_segment,
                                                                        reference_signal = ref_signal_segment,
                                                                        x_signal=x_signal_segment,
                                                                        x_ref_signal = x_ref_signal_segment,
                                                                        distance_type = distance_type,
                                                                        cost_normalized=cost_normalized,
                                                                        dDTW=dDTW)

            total_cost_matrix[node_radar_prev:node_radar+1,node_reference_prev:node_reference+1] = segment_cost_matrix

            segment_path = np.asarray(segment_path)

            #reidentification of point indices in the segment path:
            segment_path[:, 0] += node_radar_prev
            segment_path[:, 1] += node_reference_prev
            segment_path = segment_path.tolist()

            total_cost[(node_radar_prev, node_reference_prev), (node_radar, node_reference)] = segment_distance, segment_path

        cost_accumulative = 0.0
        path_accumulative = []
        path_node_pairs = []


        for k,v in total_cost.items():
            cost_accumulative+= v[0]
            path_accumulative+=v[1][0:-1]
            # peak pairs path
            path_node_pairs.append(v[1][0])
        path_accumulative.append([len(signal)-1,len(reference_signal)-1])
        path_node_pairs.append(path_accumulative[-1])
        if cost_normalized:
            cost_accumulative = cost_accumulative/np.max([len(signal),len(reference_signal)])

    return cost_accumulative, path_accumulative, path_node_pairs, total_cost_matrix


