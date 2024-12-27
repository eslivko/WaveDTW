from typing import List
import numpy as np
from src.WaveDTW import dtw_distance_wave
from src.ClassicDTW import classic_dtw_distance
from src.functions import *
from scipy.stats import pearsonr, spearmanr
from scipy.signal import resample


import time

class SimilarityEstimator():
    def __init__(self):

        self.signal = None
        self.ref_signal = None
        self.algorithm = None


    def Spearman_correlation(self, signal, signal_x_axis, ref_signal, ref_signal_x_axis):
        ref_signal_interpolated = np.interp(signal_x_axis,ref_signal_x_axis,ref_signal)
        # Calculate the Spearman correlation coefficient
        corr_spearmanr, _ = spearmanr(signal, ref_signal_interpolated)
        return np.round(corr_spearmanr,3)

    def Pearson_correlation(self, signal, signal_x_axis, ref_signal, ref_signal_x_axis):
        ref_signal_interpolated = np.interp(signal_x_axis, ref_signal_x_axis, ref_signal)
        corr_pears, _ = pearsonr(signal, ref_signal_interpolated)
        return np.round(corr_pears,3)

    def Cosine_similarity(self,signal, signal_x_axis, ref_signal, ref_signal_x_axis):
        ref_signal_interpolated = np.interp(signal_x_axis, ref_signal_x_axis, ref_signal)

        dot_product_similarity = np.dot(signal / np.linalg.norm(signal),
                                        resample(ref_signal_interpolated, len(signal)) / np.linalg.norm(
                                        resample(ref_signal_interpolated, len(signal))))
        return np.round(dot_product_similarity,3)

    def get_derivative(self, signal):

        signal_padded = np.pad(signal, (1, 0) , constant_values=(signal[0]))
        signal_padded = np.pad(signal_padded, (0, 1) , constant_values=(signal[-1]))

        signal_derivative = ((signal_padded[1:len(signal_padded) - 1] - signal_padded[0:len(signal_padded) - 2]) + (
                    signal_padded[2:len(signal_padded)] - signal_padded[0:len(signal_padded) - 2]) / 2) / 2
        return signal_derivative

    def calculate_signal_derivative(self, signal, signal_x_axis,ref_signal,ref_signal_x_axis):

        signal_derivative = self.get_derivative(signal)
        ref_signal_derivative = self.get_derivative(ref_signal)
        return signal_derivative, signal_x_axis, ref_signal_derivative, ref_signal_x_axis


    def DTW_distance(self,
                     algorithm:str,
                     signal:np.ndarray,
                     signal_x_axis:np.ndarray,
                     ref_signal:np.ndarray,
                     ref_signal_x_axis:np.ndarray,
                     distance_type = 'cityblock',
                     normalized = True,
                     use_segmentation = True):
        """
        Function estimate DTW distance and return dictionary, containing results
        :param algorithm: Type of the algorithm: 'Wave_DTW','Wave_DTW (no seg)','Wave_dDTW','Wave_dDTW (no seg)','ClassicDTW','derivativeDTW1'
        'derivativeDTW1' - is an implementation of derivative DTW based on ClassicDTW
        :param signal: 1D np.array containing 1st signal
        :param signal_x_axis: 1D np.array containing time dimension, the length should correspond to length of the 1st signal
        :param ref_signal: 1D np.array containing 2nd signal
        :param ref_signal_x_axis: 1D np.array containing time dimension, the length should correspond to length of the 2st signal
        :param distance_type: Type of the distance:
                    'absolute' (Point-to-point distance is measured based on amplitudes only),
                    'euclidean' (Point-to-pointdistance is measured based on Eclidian coordinates)
        :param normalized: bool, Set to True for normalized DTW path estimation
        :return: DTW_results: dict, containing dtw_distance, path,  dtw_picks_path (for Wave DTW algorithm), cost matrix
        """
        assert signal.shape[0]==signal_x_axis.shape[0], "Length of the signal is not equal to length of the x-axis"
        assert ref_signal.shape[0]==ref_signal_x_axis.shape[0], "Length of the reference signal is not equal to length of the x-axis"
        #print(algorithm)
        self.algorithm = algorithm
        self.dtw_distance, self.dtw_path, self.dtw_picks_path, self.total_cost_matrix = None, None, None,None
        self.DTW_results = {
         "dtw_distance": None,
         "dtw_path": None,
         "dtw_picks_path": None,
        }

        self.signal = signal
        self.signal_x_axis = signal_x_axis
        self.ref_signal = ref_signal
        self.ref_signal_x_axis = ref_signal_x_axis


        if self.algorithm=='Wave_DTW':
            start_time = time.perf_counter()


            self.signal_peaks = find_signal_peaks(self.signal)
            self.reference_peaks = find_signal_peaks(self.ref_signal)

            self.dtw_distance, self.dtw_path, self.dtw_picks_path,self.total_cost_matrix = dtw_distance_wave(signal=self.signal,
                                                                                      reference_signal=self.ref_signal,
                                                                                      signal_peaks = self.signal_peaks,
                                                                                      reference_peaks=self.reference_peaks,
                                                                                      x_signal=self.signal_x_axis,
                                                                                      x_ref_signal=self.ref_signal_x_axis,
                                                                                      distance_type=distance_type,
                                                                                      peak_matching_algorithm="min_distance",
                                                                                      cost_normalized=normalized,
                                                                                      use_segmentation = use_segmentation,
                                                                                      dDTW=False)
            end_time = time.perf_counter()
            self.duration = end_time - start_time



        if self.algorithm=='Wave_dDTW':
            start_time = time.perf_counter()

            signal_derivative, signal_derivative_x, ref_signal_derivative, ref_signal_derivative_x = self.calculate_signal_derivative(
                                                                                                                self.signal,
                                                                                                                self.signal_x_axis,
                                                                                                                self.ref_signal,
                                                                                                                self.ref_signal_x_axis)
            self.signal_peaks = find_signal_peaks(self.signal)
            self.reference_peaks = find_signal_peaks(self.ref_signal)

            self.dtw_distance, self.dtw_path, self.dtw_picks_path, self.total_cost_matrix = dtw_distance_wave(signal=signal_derivative,
                                                                                      reference_signal=ref_signal_derivative,
                                                                                      signal_peaks = self.signal_peaks,
                                                                                      reference_peaks=self.reference_peaks,
                                                                                      x_signal=signal_derivative_x,
                                                                                      x_ref_signal=ref_signal_derivative_x,
                                                                                      distance_type=distance_type,
                                                                                      peak_matching_algorithm="min_distance",
                                                                                      cost_normalized=normalized,
                                                                                      use_segmentation=use_segmentation,
                                                                                      dDTW=True)

            end_time = time.perf_counter()
            self.duration = end_time - start_time



        if self.algorithm=='ClassicDTW':
            start_time = time.perf_counter()
            self.dtw_distance, self.dtw_path, self.total_cost_matrix = classic_dtw_distance(signal=self.signal,
                                                                      reference_signal=self.ref_signal,
                                                                      x_signal=self.signal_x_axis,
                                                                      x_ref_signal=self.ref_signal_x_axis,
                                                                      distance_type=distance_type,
                                                                      cost_normalized=normalized)
            end_time = time.perf_counter()
            self.duration = end_time - start_time

        if self.algorithm == 'derivativeDTW':
            start_time = time.perf_counter()
            self.signal_derivative, self.signal_derivative_x, self.ref_signal_derivative, self.ref_signal_derivative_x = self.calculate_signal_derivative(self.signal,
                                                                                                                                                          self.signal_x_axis,
                                                                                                                                                          self.ref_signal,
                                                                                                                                                          self.ref_signal_x_axis)

            self.dtw_distance, self.dtw_path,self.total_cost_matrix = classic_dtw_distance(signal=self.signal_derivative,
                                                                      reference_signal=self.ref_signal_derivative,
                                                                      x_signal=self.signal_derivative_x,
                                                                      x_ref_signal=self.ref_signal_derivative_x,
                                                                      distance_type=distance_type,
                                                                      cost_normalized=normalized,
                                                                      dDTW=True)

            end_time = time.perf_counter()
            self.duration = end_time - start_time

        self.DTW_results["dtw_distance"] = np.round(self.dtw_distance,6)
        self.DTW_results["dtw_path"] = self.dtw_path
        self.DTW_results["dtw_picks_path"] = self.dtw_picks_path
        self.DTW_results["total_cost_matrix"] = self.total_cost_matrix

        return self.DTW_results


    def misalignment(self, path: List, signal_len:int, isDerivative = False, isWaveDTW = False):
        """
        Calculates normalized misalignment for DTW algorithm between the given path and correct path based on the number incorrect allignments.
        The correct path is given by [j,j] for j in [1,..,m]
        Implementation is based on J. Keogh et al. Derivative Dynamic Time Warping (https://epubs.siam.org/doi/epdf/10.1137/1.9781611972719.1)
        Current misalignment measure doesn't punish exceeding path
        :param path: given path of the DTW algorithm
        :param signal_len: len of the signal
        :return: M_normalized: normalized misalignment between the given path and correct path
        """
        weighted_distance = 0

        subst = 0
        # if isDerivative:
        #     subst+=2
        # if isWaveDTW:
        #     subst+=2
        distance_mask_matrix = np.zeros((signal_len-subst, signal_len-subst))
        for [i,j] in path:
            distance_mask_matrix[i,j] = 1
        for k in range(1,(signal_len-subst)//2):
            num_points = np.sum(np.diag(distance_mask_matrix,k)) + np.sum(np.diag(distance_mask_matrix,-k))

            weighted_distance+= k * num_points
        return weighted_distance/(0.5*(signal_len-subst)*(signal_len-subst-1))


    def Warpings_W(self, path: List, signal_len:int, isDerivative = False, isWaveDTW = False):
        """
        Calculates amount of warpings of the DTW path.
        E.J. Keogh et al. Derivative Dynamic Time Warping, section 4
        https://epubs.siam.org/doi/epdf/10.1137/1.9781611972719.1
        :param path: DTW path
        :param signal_len: length of the signal. it is assumed that length of both signals is equal.
        :return: W: amount of warpings in (0,1], considering that signal_len <= len(path) <= 2* signal_len - 1
        """
        subst = 0
        # if isDerivative:
        #     subst+=2
        # if isWaveDTW:
        #     subst+=2

        return (len(path)-(signal_len-subst))/(signal_len-subst)



    def estimate_all(self, signal, signal_x_axis, ref_signal, ref_signal_x_axis, distance_type='cityblock', algorithm_list = None):
        """
        Function estimates all types of similarities and returns a dictionary
        :param signal: signal (y values)
        :param signal_x_axis: signal (x values)
        :param ref_signal: reference signal (y values)
        :param ref_signal_x_axis: reference signal (x values)
        :return: similarity_estimation: List: List, containing all similarity measures.
                For each measure results are presented in a dictionary:
                dict:{
                        algorithm: str,
                        result_type:str, possible values: 'measure'/'DTW'
                        results:dict
                            {
                            'value/distance': float,
                            'path': List,
                            'path_misalignment': float,
                            'path_warpings': float,
                            'time' : float
                            }
                        }
        """

        #creation of the structure for the output results:
        if algorithm_list == None:
            algorithm_list = [
                                  ['Spearman_correlation','measure'],
                                  ['Pearson_correlation','measure'],
                                  ['Cosine_similarity','measure'],
                                  ['Classic_DTW','DTW'],
                                  ['Wave_DTW','DTW'],
                                  ['Derivative_DTW','DTW'],
                                  ['Wave_dDTW', 'DTW']

                              ]
        else:
            algorithm_type = ['measure', 'DTW']
            algorithm_list = [[algorithm , algorithm_type[1]] if 'DTW' in algorithm else [algorithm , algorithm_type[0]] for algorithm in algorithm_list]

        similarity_estimation = []
        for algorithm in algorithm_list:
            algorithm_dict = {
                                "algorithm": algorithm[0],
                                "result_type": algorithm[1],
                                "results":
                                        {
                                            "value/distance":None,
                                        }
                              }

            similarity_estimation.append(algorithm_dict)


        for algorithm_dict in similarity_estimation:

            if algorithm_dict['algorithm']=='Spearman_correlation':
                algorithm_dict['results']['value/distance'] = self.Spearman_correlation(signal=signal,
                                                                                        signal_x_axis=signal_x_axis,
                                                                                        ref_signal=ref_signal,
                                                                                        ref_signal_x_axis=ref_signal_x_axis)
            if algorithm_dict['algorithm']=='Pearson_correlation':
                algorithm_dict['results']['value/distance'] = self.Pearson_correlation(signal=signal,
                                                                                          signal_x_axis=signal_x_axis,
                                                                                          ref_signal=ref_signal,
                                                                                          ref_signal_x_axis=ref_signal_x_axis)
            if algorithm_dict['algorithm'] == 'Cosine_similarity':
                algorithm_dict['results']['value/distance'] = self.Cosine_similarity(signal=signal,
                                                                                      signal_x_axis=signal_x_axis,
                                                                                      ref_signal=ref_signal,
                                                                                      ref_signal_x_axis=ref_signal_x_axis)

            if algorithm_dict['algorithm'] == 'Classic_DTW':
                classic_DTW_results = self.DTW_distance(algorithm='ClassicDTW',
                                                        signal=signal,
                                                        signal_x_axis=signal_x_axis,
                                                        ref_signal=ref_signal,
                                                        ref_signal_x_axis=ref_signal_x_axis,
                                                        distance_type = distance_type,
                                                        normalized=True,
                                                        use_segmentation = False)
                algorithm_dict['results']['value/distance'] = classic_DTW_results["dtw_distance"]
                algorithm_dict['results']['path'] = classic_DTW_results["dtw_path"]
                algorithm_dict['results']['path_misalignment'] = self.misalignment(classic_DTW_results["dtw_path"], len(signal), isDerivative=False, isWaveDTW=False)
                algorithm_dict['results']['path_warpings'] = self.Warpings_W(classic_DTW_results["dtw_path"], len(signal), isDerivative=False, isWaveDTW=False)
                algorithm_dict['results']['time'] = self.duration

            if algorithm_dict['algorithm'].startswith('Wave_DTW'):

                use_segmantation = True
                wave_DTW_results = self.DTW_distance(algorithm=algorithm_dict['algorithm'],
                                                        signal=signal,
                                                        signal_x_axis=signal_x_axis,
                                                        ref_signal=ref_signal,
                                                        ref_signal_x_axis=ref_signal_x_axis,
                                                        distance_type = distance_type,
                                                        normalized=True,
                                                        use_segmentation = use_segmantation)
                algorithm_dict['results']['value/distance'] = wave_DTW_results["dtw_distance"]
                algorithm_dict['results']['path'] = wave_DTW_results["dtw_path"]
                algorithm_dict['results']['dtw_picks_path'] = wave_DTW_results["dtw_picks_path"]
                algorithm_dict['results']['path_misalignment'] = self.misalignment(wave_DTW_results["dtw_path"],
                                                                                   len(signal), isWaveDTW=True)
                algorithm_dict['results']['path_warpings'] = self.Warpings_W(wave_DTW_results["dtw_path"], len(signal), isWaveDTW=True)

                algorithm_dict['results']['time'] = self.duration


            if algorithm_dict['algorithm'] == 'Derivative_DTW':
                derivative_DTW_results = self.DTW_distance(algorithm='derivativeDTW',
                                                           signal=signal,
                                                           signal_x_axis=signal_x_axis,
                                                           ref_signal=ref_signal,
                                                           ref_signal_x_axis=ref_signal_x_axis,
                                                           distance_type = distance_type,
                                                           normalized=True,
                                                           use_segmentation = False)
                algorithm_dict['results']['value/distance'] = derivative_DTW_results["dtw_distance"]
                algorithm_dict['results']['path'] = derivative_DTW_results["dtw_path"]
                algorithm_dict['results']['path_misalignment'] = self.misalignment(derivative_DTW_results["dtw_path"],
                                                                                   len(signal), isDerivative=True)
                algorithm_dict['results']['path_warpings'] = self.Warpings_W(derivative_DTW_results["dtw_path"], len(signal), isDerivative=True)

                algorithm_dict['results']['time'] = self.duration

            if algorithm_dict['algorithm'].startswith('Wave_dDTW'):

                use_segmantation = True
                wave_dDTW_results = self.DTW_distance(algorithm=algorithm_dict['algorithm'],
                                                        signal=signal,
                                                        signal_x_axis=signal_x_axis,
                                                        ref_signal=ref_signal,
                                                        ref_signal_x_axis=ref_signal_x_axis,
                                                        distance_type = distance_type,
                                                        normalized=True,
                                                        use_segmentation = use_segmantation)
                algorithm_dict['results']['value/distance'] = wave_dDTW_results["dtw_distance"]
                algorithm_dict['results']['path'] = wave_dDTW_results["dtw_path"]
                algorithm_dict['results']['dtw_picks_path'] = wave_dDTW_results["dtw_picks_path"]
                algorithm_dict['results']['path_misalignment'] = self.misalignment(wave_dDTW_results["dtw_path"],
                                                                                   len(signal), isDerivative=True, isWaveDTW=True)
                algorithm_dict['results']['path_warpings'] = self.Warpings_W(wave_dDTW_results["dtw_path"], len(signal), isDerivative=True, isWaveDTW=True)
                algorithm_dict['results']['time'] = self.duration


        return similarity_estimation
