import pandas as pd
import os
import pickle
from src.signal_modifier import *
from src.similarity_estimator import *
from src.functions import max_min_scaler, find_signal_peaks
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import random
from tqdm import tqdm
import shutil
pd.set_option('display.max_columns', None)

class PhysioNet_dataset_processing():
    def __init__(self, time_column_name = "Time [s]"):
        self.root_data_folder = None
        self.save_data_folder = None
        self.dataset_list = []
        self.time_column_name = time_column_name

    def remove_files_and_dirs(self, folder_path):
        for root, dirs, files in os.walk(folder_path, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                shutil.rmtree(os.path.join(root, dir))


    def get_dataset_folder_list(self, root_dataset_folder):
        """
        Function returns list of folder names in self.root_data_folder
        :return: list of folder names in self.root_data_folder
        """
        return [folder.strip() for folder in os.listdir(root_dataset_folder) if
                   os.path.isdir(os.path.join(root_dataset_folder, folder))]


    def reformat_original_dataset(self, root_data_folder, save_data_folder, max_samples=1200):
        """
        Function loads dataset from train and test files.
        :param dataset_folder: folder name
        """
        self.root_data_folder = root_data_folder
        self.save_data_folder = save_data_folder
        self.remove_files_and_dirs(self.save_data_folder)

        self.csv_list = os.listdir(root_data_folder)
        datasets = []

        for file in self.csv_list:
            subject_id = file[:-12]
            data = pd.read_csv(os.path.join(root_data_folder, file), index_col=None)
            data.columns = [col.strip() for col in data.columns]
            num_segments = len(data) // max_samples
            print(f"File: {file} size: {data.shape[0]}")

            columns = data.columns.tolist()
            for column in columns:
                #normalization
                if column != self.time_column_name:
                    data[column] = max_min_scaler(data[column], new_min=0.0001, new_max=1)
                for i in range(num_segments):
                    #for each data segment  for each subject
                    if column != self.time_column_name:
                        self.dataset_list.append(column)
                        start_idx = i * max_samples
                        end_idx = min((i + 1) * max_samples, len(data))  # Adjust the end index for the last segment
                        segment_signal = data.iloc[start_idx:end_idx, data.columns.get_loc(column)].values
                        segment_time = data.iloc[start_idx:end_idx, data.columns.get_loc(self.time_column_name)].values
                        content = {}

                        content = {
                            self.time_column_name: segment_time,
                            'signal': segment_signal
                                   }
                        saving_folder = self.save_data_folder + column + "/"
                        if not os.path.exists(saving_folder):
                            os.makedirs(saving_folder)
                        filename = saving_folder + subject_id + "_" + str("{0:04d}".format(i)) + ".pkl"
                        with open(filename, 'wb') as f:
                            pickle.dump(content, f)
        self.dataset_list = list(set(self.dataset_list))
        print("Original datasets reformating is completed.")


    def modify_dataset_in_batch(self,raw_data_folder, modified_data_folder, parameter_dict, dataset_list=None):
        """
        Function creates modified signal for datasets from dataset_list with modification_parameters
        and saves them in modified_data_folder.

        :param raw_data_folder: root path for raw unmodified signal
        :param modified_data_folder:  root path for modified signal
        :param parameter_dict: dict containing modification parameters
        :param dataset_list: list of the datasets to modify
        :return:
        """

        if dataset_list == None:
            self.datasets_to_modify = [folder for folder in os.listdir(raw_data_folder) if
                                       os.path.isdir(os.path.join(raw_data_folder, folder))]
        else:
            self.datasets_to_modify = dataset_list

        for i, dataset in tqdm(enumerate(self.datasets_to_modify), total=len(self.datasets_to_modify)):
            self.modify_dataset(
                            raw_dataset_folder = raw_data_folder + dataset + "/",
                            modified_dataset_folder = modified_data_folder + dataset + "/",
                            parameter_dict = parameter_dict,
                            save_modified = True
                           )
        print("Datasets of modified signals has been created")


    def modify_dataset(self, raw_dataset_folder, modified_dataset_folder, parameter_dict, save_modified = True):
        """
        Function creates modified signal for dataset from given raw_dataset_folder and saves it in  the modified_dataset_folder.

        :param raw_dataset_folder: folder of raw unmodified signal
        :param modified_dataset_folder: folder for modified signal
        :param modification_parameters: dict containing modification parameters
        :param save_modified: bool
        :return:
        """
        if save_modified:
            if not os.path.exists(modified_dataset_folder):
                os.makedirs(modified_dataset_folder)

        raw_files = [file for file in os.listdir(raw_dataset_folder) if file.endswith('.pkl')]
        for i, file_name in enumerate(raw_files):
            print(file_name)
            #load data, normalize, find anchor point
            x, y = self.dataloader(raw_dataset_folder + file_name)
            x_peaks = find_signal_peaks(y)
            print("signal peaks: ", x_peaks)
            x_peaks_dist = [abs(x_peak - len(y) // 2) for x_peak in x_peaks]
            if len(x_peaks_dist)>0:
                x_anchor_id = x_peaks[np.argmin(x_peaks_dist)]
            else:
                x_anchor_id = len(y)//2

            x_modified, y_modified = self.modify_signal(x, y, x_anchor_id, parameter_dict=parameter_dict)

            #save modified
            if save_modified:
                content = {
                    "signal": y_modified,
                    self.time_column_name: x_modified
                }
                filename = modified_dataset_folder + file_name
                with open(filename, 'wb') as f:
                    pickle.dump(content, f)


    def dataloader(self, filename: str, signal_field_name = "signal"):
        file = open(filename, "rb")
        content = pickle.load(file)
        raw_signal = content[signal_field_name]
        time_label = content[self.time_column_name]

        return time_label, raw_signal

    def modify_signal(self, x, y, x_anchor_id, parameter_dict):

        signal_modifier = SignalModifier(time_warping=True, gaussian_bump=True)

        x_modified, y_modified, y_interpolated = signal_modifier.modify(x=x,
                                                                        y=y,
                                                                        x_anchor_id=x_anchor_id,
                                                                        time_warping_window=parameter_dict[
                                                                            'time_warping_window'],  # 3.0,
                                                                        time_warping_rate=parameter_dict[
                                                                            'time_warping_rate'],  # 2.0,
                                                                        bump_window=parameter_dict['bump_window'],
                                                                        # np.pi / 3,
                                                                        bump_height=parameter_dict[
                                                                            'bump_height'])  # 0.5)


        return x_modified, y_modified

    def find_signal_peaks(self, signal, peaks_distance=12, peaks_height=0.03, prominence=0.04):

        peaks = find_peaks(x=signal,
                           height=peaks_height,
                           distance=peaks_distance,
                           prominence=prominence)[0]
        return peaks

    def compare_signals_in_batch(self,
                                 raw_data_folder,
                                 modified_data_folder,
                                 result_data_folder,
                                 img_data_folder,
                                 distance_type,
                                 algorithm_list,
                                 save_results = True,
                                 dataset_list=None):


        if dataset_list is None:
            dataset_folder_list = [folder for folder in os.listdir(raw_data_folder) if os.path.isdir(os.path.join(raw_data_folder, folder))]
        else:
            dataset_folder_list = dataset_list


        for i, dataset in enumerate(dataset_folder_list):
            print("Signal in process: ", dataset)
            raw_dataset_folder = raw_data_folder +  dataset + "/"
            modified_dataset_folder = modified_data_folder + dataset + "/"

            if not os.path.exists(result_data_folder):
                os.mkdir(result_data_folder)
            if not os.path.exists(img_data_folder):
                os.mkdir(img_data_folder)

            result_dataset_folder = result_data_folder + dataset + "/"
            if not os.path.exists(result_dataset_folder):
                os.mkdir(result_dataset_folder)
            img_dataset_folder = img_data_folder + dataset + "/"
            if not os.path.exists(img_dataset_folder):
                os.mkdir(img_dataset_folder)

            self.compare_signals(dataset = dataset,
                                 raw_dataset_folder=raw_dataset_folder,
                                 modified_dataset_folder=modified_dataset_folder,
                                 result_dataset_folder = result_dataset_folder,
                                 img_dataset_folder = img_dataset_folder,
                                 distance_type = distance_type,
                                 algorithm_list = algorithm_list,
                                 save_results=save_results
                                 )



    def compare_signals(self,
                        dataset,
                        raw_dataset_folder,
                        modified_dataset_folder,
                        result_dataset_folder,
                        img_dataset_folder,
                        distance_type,
                        algorithm_list,
                        save_results = True):

        column_names = ['algorithm',
                        'time_warping_window',
                        'time_warping_rate',
                        'bump_window',
                        'bump_height',
                        'value/distance',
                        'user_session',
                        'path',
                        'path_misalignment',
                        'path_warpings',
                        'time']


        df_rows_list = []

        raw_files = [file for file in os.listdir(raw_dataset_folder) if file.endswith('.pkl')]

        for j, file_name in tqdm(enumerate(raw_files), total=len(raw_files)):


            session_id = file_name.split('.')[0]
            #load data and compare
            x, y = self.dataloader(raw_dataset_folder + file_name)
            x_modified, y_modified = self.dataloader(modified_dataset_folder + file_name)


            similarity_results = self.similarity_estimation(y=y, x=x, y_modified=y_modified,
                                                       x_modified=x_modified, distance_type=distance_type, algorithm_list = algorithm_list)

            for algorithm_results in similarity_results:

                if algorithm_list is None or algorithm_results['algorithm'] in algorithm_list:
                    row_dict_detailed = parameter_dict.copy()
                    row_dict_detailed['user_session'] = session_id
                    row_dict_detailed['algorithm'] = algorithm_results['algorithm']

                    algorithm_results_dict = algorithm_results['results']
                    for k, v in algorithm_results_dict.items():
                        row_dict_detailed[k] = v

                    df_rows_list.append(row_dict_detailed)

                    peak_path = None
                    if algorithm_results['algorithm'] == "Wave_DTW" or algorithm_results['algorithm'] == "Wave_dDTW":

                        peak_path = algorithm_results['results']["dtw_picks_path"]


                    if algorithm_results['result_type'] == 'DTW':
                        dtw_path = row_dict_detailed['path']
                        savepath = img_dataset_folder + row_dict_detailed['user_session'] + '_'
                        parameter_str = row_dict_detailed['user_session'].split('_')
                        parameter_str = 'Signal ' + parameter_str[0] + "_" + parameter_str[1] + ' (part ' + parameter_str[2] + ')'

                        self.visualize_dtw_path(signal=y, signal_x_axis=x, ref_signal=y_modified, ref_signal_x_axis=x_modified,
                                     path=dtw_path, dtw_picks_path=peak_path, algorithm=algorithm_results['algorithm'],
                                     parameter_str=parameter_str, savefig=save_results, savepath=savepath,
                                     extensions=['pdf', 'png'])


        df = pd.DataFrame(df_rows_list, columns=column_names)
        if save_results:
            result_dataset_path = result_dataset_folder + dataset + ".txt"
            df.to_csv(result_dataset_path, sep='|')
        print('\nCompleted!')
        return df


    def similarity_estimation(self, y, x, y_modified, x_modified, distance_type, algorithm_list):
        similarity_estimator = SimilarityEstimator()
        similarity_results = similarity_estimator.estimate_all(y, x, y_modified, x_modified, distance_type = distance_type, algorithm_list = algorithm_list)
        return similarity_results

    def visualize_dtw_path(self, signal, signal_x_axis, ref_signal, ref_signal_x_axis, path, dtw_picks_path, algorithm='',
                     parameter_str='', savefig=False, savepath='', extensions=[]):

        label_dict = {'Classic_DTW': 'DTW', 'Wave_DTW': 'Wave DTW', 'Derivative_DTW': 'Derivative DTW', 'Wave_dDTW': 'Wave derivative DTW'}

        divider = [2.6,3,4,5,1]

        path_sample = []
        divider_id  = 0


        while path_sample == [] and divider_id < len(divider):

            if divider[divider_id] > 1:
                start_id = int(len(signal) // divider[divider_id])
                end_id = len(signal) - int( len(signal) // divider[divider_id])
            else:
                start_id = 0
                end_id = len(signal) - 1

            signal_sample, signal_x_axis_sample = signal[start_id:end_id], signal_x_axis[start_id:end_id]
            ref_signal_sample, ref_signal_x_axis_sample = ref_signal[start_id:end_id], ref_signal_x_axis[start_id:end_id]

            path_sample = [[i, j] for [i, j] in path if
                           (i >= start_id) and (i <= end_id) and (j >= start_id) and (j <= end_id)]
            divider_id +=1

        plt.figure(figsize=(8, 8))


        plt.plot(signal_x_axis_sample, signal_sample, c='#027aba', linewidth=1.8,
                 label='Original signal')

        plt.plot(ref_signal_x_axis_sample, ref_signal_sample, c='#ba025e', linewidth=1.8,
                 label='Modified signal')
        dtw_x_coordinates = np.asarray([[signal_x_axis[i], ref_signal_x_axis[j]] for [i, j] in path_sample])
        dtw_y_coordinates = np.asarray([[signal[i], ref_signal[j]] for [i, j] in path_sample])

        plt.plot(dtw_x_coordinates[0], dtw_y_coordinates[0], '-', alpha=0.9, linewidth=0.8,
                 color='grey', label="DTW path")
        for i in range(1, len(dtw_x_coordinates), 2):
            plt.plot(dtw_x_coordinates[i], dtw_y_coordinates[i], '-', alpha=0.9, linewidth=0.8,
                     color='grey')

        if dtw_picks_path:
            dtw_picks_path_sample = [[i, j] for [i, j] in dtw_picks_path if
                                     (i >= start_id) and (i <= end_id) and (j >= start_id) and (j <= end_id)]
            dtw_peak_pairs_x_coordinates = np.asarray(
                [[signal_x_axis[i], ref_signal_x_axis[j]] for [i, j] in dtw_picks_path_sample])
            dtw_peak_pairs_y_coordinates = np.asarray(
                [[signal[i], ref_signal[j]] for [i, j] in dtw_picks_path_sample])

            for i in range(1, len(dtw_picks_path_sample), 1):
                plt.plot(dtw_peak_pairs_x_coordinates[i], dtw_peak_pairs_y_coordinates[i], '-', alpha=0.8,
                         color='#555555', linewidth=1.2)

        plt.legend(fontsize=20)
        plt.xlabel('Time',fontsize=20)
        plt.ylabel('Amplitude',fontsize=20)

        algorithm_display = label_dict[algorithm]
        plt.title(f'{algorithm_display} path',fontsize=20)
        plt.grid(True)
        if savefig:
            for ext in extensions:
                filename = savepath + algorithm + '.' + ext
                plt.savefig(filename, bbox_inches='tight')
            plt.close()
        # plt.show()






orig_data_folder = "./data/data_BIDMC_Resp_Signals/"
raw_data_folder = "./data/data_BIDMC_Resp_600/raw/"
modified_data_folder = "./data/data_BIDMC_Resp_600/modified/"
result_data_folder = "./data/results_BIDMC_Resp/"
img_data_folder = "./data/img_BIDMC_Resp/"

parameter_dict = {
    'time_warping_window': 0.25,
    'time_warping_rate': 1.0,
    'bump_window': 1.0,
    'bump_height': 0.5
     }

PhysioNet_dataset_processor = PhysioNet_dataset_processing()


PhysioNet_dataset_processor.reformat_original_dataset(root_data_folder=orig_data_folder, save_data_folder=raw_data_folder, max_samples=600)
PhysioNet_dataset_processor.modify_dataset_in_batch(raw_data_folder = raw_data_folder,
                                                    modified_data_folder = modified_data_folder,
                                                    parameter_dict= parameter_dict,
                                                    dataset_list = None)
dataset_list = PhysioNet_dataset_processor.dataset_list
print("Signals: ", dataset_list)

distance_type='cityblock'

algorithm_list = ['Classic_DTW','Wave_DTW', 'Derivative_DTW', 'Wave_dDTW']


PhysioNet_dataset_processor.compare_signals_in_batch(
                                 raw_data_folder = raw_data_folder,
                                 modified_data_folder = modified_data_folder,
                                 result_data_folder = result_data_folder,
                                 img_data_folder = img_data_folder,
                                 distance_type = distance_type,
                                 algorithm_list = algorithm_list,
                                 save_results = True,
                                 dataset_list=None)


