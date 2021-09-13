#! /usr/bin/env python3
'''Estimates spatio-temporal gait parameters from the raw 
trajectories of reflective markers and a file with 
the definition of relevant events during capture.'''
import ast
from pathlib import Path
import pandas as pd
import numpy as np
from utils import *
from scipy.signal import savgol_filter

class SpatioTemporalEstimator:
    '''Class to handle all spatio-temporal parameter estimation'''

    def __init__(self, dataset_path, records, filtering):
        self.dataset = dataset_path
        self.records = records
        self.filtering_path = filtering
        self.turn_list = []
        self.errors = []
        self.stride_filtering = None
        self.spatiotemporal_data = {}
        self.anthropometry = self.load_anthropometry_data()

    def load_anthropometry_data(self):
        '''Load file with anthropometric characteristics'''
        path = Path(self.dataset)
        anthropometry_path = str(
            path.parent.absolute()) + "/anthropometry.xlsx"
        return pd.read_excel(anthropometry_path, skipfooter=1)

    def load_stride_filtering(self, add_spatitemporal=True,):
        '''Load file with stride filtering'''
        self.stride_filtering = pd.read_excel(self.filtering_path)

        if add_spatitemporal:
            self.stride_filtering["Spatiotemporal_Left_Strides_Filtering"] = np.empty(
                (len(self.stride_filtering), 0)).tolist()
            self.stride_filtering["Spatiotemporal_Right_Strides_Filtering"] = np.empty(
                (len(self.stride_filtering), 0)).tolist()

    def load_raw_data(self, path, frame_init=0, frame_end=3000):
        '''Load file with marker trajectories'''
        # Read used markers (header)
        with open(path) as f:
            markers_line = f.readline()
        markers = markers_line.split('\t')[:-1]

        # Prepare column names
        headers = ['frame', 'time']

        for mrk in markers:
            headers.append(mrk + "_PX")
            headers.append(mrk + "_PY")
            headers.append(mrk + "_PZ")
            headers.append(mrk + "_VX")
            headers.append(mrk + "_VY")
            headers.append(mrk + "_VZ")
            headers.append(mrk + "_AX")
            headers.append(mrk + "_AY")
            headers.append(mrk + "_AZ")

        # Load file using column names
        raw = pd.read_csv(path, sep='\t', lineterminator='\n', header=None, skiprows=[0],
                          usecols=range(len(headers)), names=headers)

        # Interpolate missing points
        for mrk in headers[2:]:
            # Comprueba si tiene valores nulos
            if not raw[raw[mrk].isnull()].empty:
                raw[mrk] = raw[mrk].interpolate('polynomial', order=2)

        # Apply filtering for noise reduction
        time = raw['time']
        frame = raw['frame']

        raw = raw.loc[:, ~raw.columns.isin(['time', 'frame'])].apply(lambda col: apply_savgol_filter(col))
        raw["time"] = time
        raw["frame"] = frame

        return raw

    def stride_length(self, stride, foot, raw):
        '''Calculate stride length using the heel marker and the events file.'''
        # Foot Initial Contact 1
        start = stride[0][2]
        # Foot Initial Contact 2
        end = stride[len(stride) - 1][2]

        if not np.isnan(start) and not np.isnan(end):
            start = int(round(start * 100))
            end = int(round(end * 100))
            pos_init = raw[foot + "_heel_PZ"][start]
            pos_end = raw[foot + "_heel_PZ"][end]
            return abs(pos_end - pos_init)

        return np.NaN

    def step_length(self, stride, foot, raw):
        '''Calculate step length using the heel marker and the events file.'''
        # Opposite Foot Initial Contact
        start = get_event_time(stride, 'EVENT_' + opposite_foot(foot.upper()) +
                               '_FOOT_INITIAL_CONTACT')
        # Foot Initial Contact 2
        end = stride[len(stride) - 1][2]

        if not np.isnan(start) and not np.isnan(end):
            start = int(round(start * 100))
            end = int(round(end * 100))
            pos_init = raw[foot + "_heel_PZ"][start]
            pos_end = raw[opposite_foot(foot).lower() + "_heel_PZ"][end]
            return abs(pos_end - pos_init)

        return np.NaN

    def stride_heel_height(self, stride, foot, raw):
        '''TODO'''
        start = stride[0][2]  # Foot Initial Contact 1
        end = stride[len(stride) - 1][2]  # Foot Initial Contact 2

        if not np.isnan(start) and not np.isnan(end):
            start = int(round(start * 100))
            end = int(round(end * 100))
            result = (raw[foot + "_heel_PY"][start:end + 10].max())
            return result if result > 0 else 0

        return np.NaN

    def stride_toe_height(self, stride, foot, raw):
        '''TODO'''
        start = stride[0][2]  # Foot Initial Contact 1
        end = stride[len(stride) - 1][2]  # Foot Initial Contact 2

        if not np.isnan(start) and not np.isnan(end):
            start = int(round(start * 100))
            end = int(round(end * 100))
            result = (raw[foot + "_toe_PY"][start:end + 10].max())
            return result if result > 0 else 0

        return np.NaN

    def base_of_support(self, stride, raw):
        '''TODO'''
        start = stride[0][2]  # Foot Initial Contact 1
        end = stride[len(stride) - 1][2]  # Foot Initial Contact 2

        if not np.isnan(start) and not np.isnan(end):
            start = int(round(start * 100))
            end = int(round(end * 100))
            pos_left = raw["left_heel_PX"][start: end].values
            pos_right = raw["right_heel_PX"][start: end].values
            result = abs(pos_left - pos_right).mean()
            return result if result > 0 else 0

        return np.NaN

    def step_angle(self, time_step1, time_step2, foot, raw):
        '''TODO'''
        if np.isnan(time_step1) or np.isnan(time_step2):
            return np.NaN

        time_step1 = int(round(time_step1 * 100))
        time_step2 = int(round(time_step2 * 100))

        # Define points
        heel1_x = raw[foot + "_heel_PX"][time_step1]
        heel1_z = raw[foot + "_heel_PZ"][time_step1]
        heel2_x = raw[foot + "_heel_PX"][time_step2]
        heel2_z = raw[foot + "_heel_PZ"][time_step2]
        toe_x = raw[foot + "_toe_PX"][time_step2]
        toe_z = raw[foot + "_toe_PZ"][time_step2]

        # Distance between points
        d_heel1_heel2 = distance((heel1_x, heel1_z), (heel2_x, heel2_z))
        d_heel1_toe = distance((heel1_x, heel1_z), (toe_x, toe_z))
        d_heel2_toe = distance((heel2_x, heel2_z), (toe_x, toe_z))

        beta = math.degrees(math.acos(((d_heel1_heel2**2) + (d_heel2_toe**2) - (d_heel1_toe**2))
                                      / (2 * d_heel1_heel2 * d_heel2_toe)))

        return 180 - beta

    def com_vertical_displacement(self, time_start, time_end, cdg):
        '''TODO'''
        if np.isnan(time_start) or np.isnan(time_end):
            return np.NaN

        time_start = int(round(time_start * 100))
        time_end = int(round(time_end * 100))

        min_cdg = cdg['CDG Y'][time_start:time_end].min()
        max_cdg = cdg['CDG Y'][time_start:time_end].max()

        return abs(min_cdg - max_cdg) * 10

    def com_horizontal_displacement(self, time_start, time_end, cdg):
        '''TODO'''
        if np.isnan(time_start) or np.isnan(time_end):
            return np.NaN

        time_start = int(round(time_start * 100))
        time_end = int(round(time_end * 100))

        min_cdg = cdg['CDG X'][time_start:time_end].min()
        max_cdg = cdg['CDG X'][time_start:time_end].max()

        return abs(min_cdg - max_cdg) / 2 * 10

    def toeoff_angle(self, toeoff_time, foot, raw):
        '''TODO'''
        toeoff_time = int(round(toeoff_time * 100))

        # Define puntos
        heel_y = raw[foot + "_heel_PY"][toeoff_time]
        heel_z = raw[foot + "_heel_PZ"][toeoff_time]
        toe_y = raw[foot + "_toe_PY"][toeoff_time]
        toe_z = raw[foot + "_toe_PZ"][toeoff_time]
        aux_y = toe_y
        aux_z = heel_z

        # Calculate distances
        d_heel_toe = distance((heel_z, heel_y), (toe_z, toe_y))
        d_toe_aux = distance((toe_z, toe_y), (aux_z, aux_y))
        d_heel_aux = distance((heel_z, heel_y), (aux_z, aux_y))

        return math.degrees(math.acos(((d_heel_toe**2) + (d_toe_aux**2) -
                                       (d_heel_aux**2)) / (2 * d_heel_toe * d_toe_aux)))

    def strike_angle(self, stride_init, foot, raw):
        '''TODO'''
        # Define points
        stride_init = int(round(stride_init * 100))
        heel_y = raw[foot + "_heel_PY"][stride_init]
        heel_z = raw[foot + "_heel_PZ"][stride_init]
        toe_y = raw[foot + "_toe_PY"][stride_init]
        toe_z = raw[foot + "_toe_PZ"][stride_init]
        aux_y = heel_y
        aux_z = toe_z

        # Calculate distances
        d_heel_toe = distance((heel_z, heel_y), (toe_z, toe_y))
        d_toe_aux = distance((toe_z, toe_y), (aux_z, aux_y))
        d_heel_aux = distance((heel_z, heel_y), (aux_z, aux_y))

        return math.degrees(math.acos(((d_heel_toe**2) + (d_toe_aux**2)
                                       - (d_heel_aux**2)) / (2 * d_heel_toe * d_toe_aux)))

    def spatial_parameters(self, stride_events, foot, raw, cdg):
        '''TODO'''
        foot = foot.lower()
        op_foot = opposite_foot(foot).lower()

        # Prepare dataframe
        df = pd.DataFrame([0 for stride in stride_events],
                          columns=["stride_length", ])

        df.index.name = 'stride'

        # initial contact 1 (stride[0][2]) y initial contact 2 (stride[4][2])
        df["stride_length"] = list(
            map(lambda stride: self.stride_length(stride, foot, raw), stride_events))
        df["step_length"] = list(
            map(lambda stride: self.step_length(stride, foot, raw), stride_events))
        df["max_heel_height"] = list(
            map(lambda stride: self.stride_heel_height(stride, foot, raw), stride_events))
        df["base_of_support"] = list(
            map(lambda stride: self.base_of_support(stride, raw), stride_events))
        df["max_toe_height"] = list(
            map(lambda stride: self.stride_toe_height(stride, foot, raw), stride_events))

        step_angle_times = list(map(lambda item: (np.NaN, item[1][0][2]) if item[0] == 0
                                    else (stride_events[item[0] - 1][0][2], item[1][0][2]),
                                    list(enumerate(stride_events))))
        df["step_angle"] = list(map(lambda item: self.step_angle(
            item[0], item[1], foot, raw), step_angle_times))

        df["strike_angle"] = list(
            map(lambda stride: self.strike_angle(stride[0][2], foot, raw), stride_events))
        df["toe_off_angle"] = list(
            map(lambda stride: self.toeoff_angle(stride[0][2], foot, raw), stride_events))

        # Desplazamiento CDG
        df["com_vertical_displacement"] = list(map(lambda stride:
                                                   self.com_vertical_displacement(stride[0][2],
                                                                                  stride[len(
                                                                                      stride) - 1][2],
                                                                                  cdg),
                                                   stride_events))
        df["com_horizontal_displacement"] = list(map(lambda stride:
                                                     self.com_horizontal_displacement(
                                                         stride[0][2], stride[len(stride) - 1][2], cdg),
                                                     stride_events))

        # Start row index from 1
        df.index += 1
        df.round(2)
        return df

    def temporal_parameters(self, stride_events, foot):
        '''TODO'''
        foot = foot.lower()
        op_foot = opposite_foot(foot).lower()

        # Prepare dataframe
        df = pd.DataFrame([stride[0][2] for stride in stride_events], columns=[
                          (foot + '_initial_contact_1')])
        df[(op_foot + '_toe_off')] = list(map(lambda stride: get_event_time(stride,
                                                                            'EVENT_' + op_foot.upper()
                                                                            + '_FOOT_TOE_OFF'), stride_events))
        df[(op_foot + '_initial_contact')] = list(map(lambda stride: get_event_time(stride,
                                                                                    'EVENT_'
                                                                                    + op_foot.upper()
                                                                                    + '_FOOT_INITIAL_CONTACT'),
                                                      stride_events))
        df[(foot + '_toe_off')] = list(map(lambda stride: get_event_time(stride,
                                                                         'EVENT_'
                                                                         + foot.upper() + '_FOOT_TOE_OFF'),
                                           stride_events))
        df[(foot + '_initial_contact_2')] = [stride[len(stride) - 1][2]
                                             for stride in stride_events]

        df.index.name = 'stride'
        df.index += 1

        # Estimate temporal parameters
        df['stride_duration'] = df[(foot + '_initial_contact_2')] - \
            df[(foot + '_initial_contact_1')]
        df['step_duration'] = df[(op_foot + '_initial_contact')] - \
            df[(foot + '_initial_contact_1')]
        df['stance_duration'] = df[(foot + '_toe_off')] - \
            df[(foot + '_initial_contact_1')]
        df['swing_duration'] = df[(foot + '_initial_contact_2')
                                  ] - df[(foot + '_toe_off')]
        df['stance_percent'] = (df['stance_duration'] /
                                df['stride_duration']) * 100
        df['swing_percent'] = (df['swing_duration'] /
                               df['stride_duration']) * 100

        df["loading_end"] = ((df[op_foot + "_toe_off"] - df[foot +
                             "_initial_contact_1"]) / df["stride_duration"]) * 100
        df["preswing_init"] = ((df[op_foot + "_initial_contact"] -
                               df[foot + "_initial_contact_1"]) / df["stride_duration"]) * 100

        df['double_support1_duration'] = df[(
            op_foot + '_toe_off')] - df[(foot + '_initial_contact_1')]
        df['double_support2_duration'] = df[(
            foot + '_toe_off')] - df[(op_foot + '_initial_contact')]
        df['double_support_duration'] = df['double_support1_duration'] + \
            df['double_support2_duration']

        df['double_support_percent'] = (
            df['double_support_duration'] / df['stride_duration']) * 100

        # Start row index from 1
        # df.index = df.index + 1

        df.round(2)
        return df

    def estimate_gait_time(self, left_strides_temporal_df, right_strides_temporal_df):
        '''TODO'''
        gait_start = min([left_strides_temporal_df.left_initial_contact_1.iloc[0],
                         right_strides_temporal_df.right_initial_contact_1.iloc[0]])

        gait_end = max([left_strides_temporal_df.left_initial_contact_2.max(
        ), right_strides_temporal_df.right_initial_contact_2.max()])

        return gait_end - gait_start

    def estimate_speed_cadence(self, right_stride_duration, left_stride_duration, right_stride_length, left_stride_length):
        '''TODO'''
        stride_duration = (right_stride_duration + left_stride_duration) / 2
        stride_length = (right_stride_length + left_stride_length) / 2

        speed = (stride_length / 1000) / stride_duration
        cadence = (speed / stride_length) * 120 * 1000

        return speed, cadence

    def normalize_spatiotemporal(self, overview, subject_id):
        '''TODO'''
        norm_overview = {}
        leg_length = self.anthropometry.loc[self.anthropometry.subject ==
                                            subject_id, "leg_length"].iloc[0]
        foot_length = self.anthropometry.loc[self.anthropometry.subject ==
                                             subject_id, "foot_length"].iloc[0]

        norm_overview["Right_Stride_Length"] = overview["Right_Stride_Length"] / leg_length
        norm_overview["Left_Stride_Length"] = overview["Left_Stride_Length"] / leg_length
        norm_overview["Right_Step_Length"] = overview["Right_Step_Length"] / leg_length
        norm_overview["Left_Step_Length"] = overview["Left_Step_Length"] / leg_length
        norm_overview["Base_Of_Support"] = overview["Base_Of_Support"] / leg_length
        norm_overview["Right_Heel_Height"] = overview["Right_Heel_Height"] / foot_length
        norm_overview["Left_Heel_Height"] = overview["Left_Heel_Height"] / foot_length
        norm_overview["Right_Toe_Height"] = overview["Right_Toe_Height"] / foot_length
        norm_overview["Left_Toe_Height"] = overview["Left_Toe_Height"] / foot_length
        norm_overview["Com_Vertical_Displacement"] = overview["Com_Vertical_Displacement"] / foot_length
        norm_overview["Com_Horizontal_Displacement"] = overview["Com_Horizontal_Displacement"] / leg_length
        norm_overview["Speed"] = overview["Speed"] / \
            (math.sqrt(9.806 * leg_length))
        norm_overview["Cadence"] = overview["Cadence"] / \
            (math.sqrt(9.806 / leg_length))
        norm_overview["Right_Stance_Phase_Duration"] = overview["Right_Stance_Phase_Duration"] / \
            overview["Right_Stride_Duration"]
        norm_overview["Right_Swing_Phase_Duration"] = overview["Right_Swing_Phase_Duration"] / \
            overview["Right_Stride_Duration"]
        norm_overview["Left_Stance_Phase_Duration"] = overview["Left_Stance_Phase_Duration"] / \
            overview["Left_Stride_Duration"]
        norm_overview["Left_Swing_Phase_Duration"] = overview["Left_Swing_Phase_Duration"] / \
            overview["Left_Stride_Duration"]
        norm_overview["Right_Double_Support_Duration"] = overview["Right_Double_Support_Duration"] / \
            overview["Right_Stride_Duration"]
        norm_overview["Left_Double_Support_Duration"] = overview["Left_Double_Support_Duration"] / \
            overview["Left_Stride_Duration"]
        norm_overview["Left_Step_Duration"] = overview["Left_Step_Duration"] / \
            overview["Left_Stride_Duration"]
        norm_overview["Right_Step_Duration"] = overview["Right_Step_Duration"] / \
            overview["Right_Stride_Duration"]

        return norm_overview

    def write_output(self, subject_id, record_id):
        '''TODO'''
        record_data = {'subject' : subject_id, 'record': record_id}
        output_path = os.path.join(self.dataset, subject_id, record_id, "preprocessed", (
            subject_id + "_" + record_id + ".spatiotemporal.xlsx"))

        spatiotemporal = self.spatiotemporal_data[subject_id + "_" + record_id]
        right_strides_temporal = spatiotemporal['temporal']['right']
        left_strides_temporal = spatiotemporal['temporal']['left']
        right_strides_spatial = spatiotemporal['spatial']['right']
        left_strides_spatial = spatiotemporal['spatial']['left']

        right_means = pd.concat([right_strides_temporal.mean().round(
            2), right_strides_spatial.mean().round(2)])
        left_means = pd.concat([left_strides_temporal.mean().round(
            2), left_strides_spatial.mean().round(2)])

        gait_time = self.estimate_gait_time(
            left_strides_temporal, right_strides_temporal)

        filtering = self.stride_filtering.loc[(self.stride_filtering['Record'] == record_id) & (
            self.stride_filtering['Subject'] == subject_id)]
        left_strides_discarded = len(ast.literal_eval(
            filtering['Left_Strides_Filtering'].values.tolist()[0]))
        right_strides_discarded = len(ast.literal_eval(
            filtering['Right_Strides_Filtering'].values.tolist()[0]))

        speed, cadence = self.estimate_speed_cadence(
            right_means['stride_duration'], left_means['stride_duration'], right_means['stride_length'], left_means['stride_length'])

        overview = {
            'Total_Strides_Left': len(left_strides_temporal) + left_strides_discarded,
            'Total_Strides_Right': len(right_strides_temporal) + right_strides_discarded,
            'Processed_Strides_Left': len(left_strides_temporal),
            'Processed_Strides_Right': len(right_strides_temporal),
            'Turns': filtering['Turns_Count'].values.tolist()[0],
            'Gait_Time': gait_time,
            'Cadence': cadence,
            'Speed': speed,
            'Right_Stance_Phase_Duration': right_means['stance_duration'],
            'Right_Stance_Percent': right_means['stance_percent'],
            'Right_Swing_Phase_Duration': right_means['swing_duration'],
            'Right_Swing_Phase_Percent': right_means['swing_percent'],
            'Left_Stance_Phase_Duration': left_means['stance_duration'],
            'Left_Stance_Percent': left_means['stance_percent'],
            'Left_Swing_Phase_Duration': left_means['swing_duration'],
            'Left_Swing_Phase_Percent': left_means['swing_percent'],
            'Right_Stride_Duration': right_means['stride_duration'],
            'Left_Stride_Duration': left_means['stride_duration'],
            'Right_Step_Duration': right_means['step_duration'],
            'Left_Step_Duration': left_means['step_duration'],
            'Right_Double_Support_Duration': (right_means['double_support_duration']),
            'Left_Double_Support_Duration': (left_means['double_support_duration']),
            'Double_Support_Duration': ((left_means['double_support_duration'] + right_means['double_support_duration']) / 2).round(2),
            'Right_Double_Support_Percent': (right_means['double_support_percent']),
            'Left_Double_Support_Percent': (left_means['double_support_percent']),
            'Double_Support_Percent': ((left_means['double_support_percent'] + right_means['double_support_percent']) / 2).round(2),
            'Right_Stride_Length': right_means['stride_length'],
            'Left_Stride_Length': left_means['stride_length'],
            'Stride_Length': (right_means['stride_length'] + left_means['stride_length']) / 2,
            'Right_Step_Length': right_means['step_length'],
            'Left_Step_Length': left_means['step_length'],
            'Step_Length': (right_means['step_length'] + left_means['step_length']) / 2,
            'Base_Of_Support': ((left_means['base_of_support'] + right_means['base_of_support']) / 2).round(2),
            'Right_Heel_Height': right_means['max_heel_height'],
            'Left_Heel_Height': left_means['max_heel_height'],
            'Heel_Height': (right_means['max_heel_height'] + left_means['max_heel_height']) / 2,
            'Right_Step_Angle': right_means['step_angle'],
            'Left_Step_Angle': left_means['step_angle'],
            'Right_Toe_Off_Angle': right_means['toe_off_angle'],
            'Left_Toe_Off_Angle': left_means['toe_off_angle'],
            'Right_Strike_Angle': right_means['strike_angle'],
            'Left_Strike_Angle': left_means['strike_angle'],
            'Right_Toe_Height': right_means['max_toe_height'],
            'Left_Toe_Height': left_means['max_toe_height'],
            'Toe_Height': (right_means['max_toe_height'] + left_means['max_toe_height']) / 2,
            'Com_Vertical_Displacement': (right_means['com_vertical_displacement'] + left_means['com_vertical_displacement']) / 2,
            'Com_Horizontal_Displacement': (right_means['com_horizontal_displacement'] + left_means['com_horizontal_displacement']) / 2,
        }

        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(output_path, engine='openpyxl')

        # Write workseet from overview map
        write_sheet_from_dict(writer, 'Overview', overview)

        # Normalize spatiotemporal parameters
        overview_norm = self.normalize_spatiotemporal(overview, subject_id)
        write_sheet_from_dict(writer, 'Norm-Overview', overview_norm)

        # Write each dataframe to a different worksheet.
        write_sheet_from_df(writer, 'Right_Gait_Cycle',
                            right_strides_temporal)
        write_sheet_from_df(writer, 'Left_Gait_Cycle', left_strides_temporal)

        write_sheet_from_df(writer, 'Right_Spatial',
                            right_strides_spatial)
        write_sheet_from_df(writer, 'Left_Spatial', left_strides_spatial)

        # Close the Pandas Excel writer and output the Excel file.
        writer.save()

        # Prepare data to generate global CSV
        exclude_values = ['Total_Strides_Left', 'Total_Strides_Right', 'Processed_Strides_Left',
         'Processed_Strides_Right', 'Turns', 'Stride_Length', 'Step_Length', 
         'Heel_Height', 'Toe_Height', 'Double_Support_Percent', 'Double_Support_Duration']
        for key in overview_norm:
            record_data[key.lower() + "_norm"] = overview_norm[key]

        for key in overview:
            if key not in overview_norm.keys() and key not in exclude_values:
                record_data[key.lower()] = overview[key]


        return record_data

    def find_stride_outliers(self, strides_spatial_df, subject, record, foot):
        '''TODO'''
        nan_values = strides_spatial_df[strides_spatial_df['stride_length'].isnull(
        )].index.tolist()
        filtering = self.stride_filtering.loc[(self.stride_filtering['Record'] == record) & (
            self.stride_filtering['Subject'] == subject)]
        turns_filtering = ast.literal_eval(
            filtering['Turns_' + (foot.title()) + '_Strides_Filtering'].values.tolist()[0])
        mean = strides_spatial_df['stride_length'].drop(
            turns_filtering, axis=0).mean()
        outliers = strides_spatial_df[strides_spatial_df['stride_length']
                                      < mean * 0.7].index.tolist()

        step_angle_outliers = strides_spatial_df[strides_spatial_df['step_angle']
                                                 > 90].index.tolist()

        # Add to table
        index = self.stride_filtering.loc[(self.stride_filtering["Subject"] == subject)
                                          & (self.stride_filtering["Record"] == record)].index[0]

        self.stride_filtering.at[index, 'Spatiotemporal_' + (
            foot.title()) + '_Strides_Filtering'] = list(set(nan_values + outliers + step_angle_outliers))

    def process_record(self, dataset, subject_id, record_id):
        '''TODO'''
        data_path = os.path.join(dataset, subject_id, record_id)
        self.spatiotemporal_data[subject_id + "_" + record_id] = {'temporal': {
            'left': None, 'right': None}, 'spatial': {'left': None, 'right': None}}

        # Load events file
        events_path = os.path.join(
            data_path, (subject_id + "_" + record_id + ".events.txt"))
        events = load_events(events_path)

        # Split events by strides
        right_events = split_stride(events, 'RIGHT')
        left_events = split_stride(events, 'LEFT')

        # Load raw data
        raw_path = os.path.join(
            data_path, (subject_id + "_" + record_id + ".raw"))
        hs_events = [event for event in events if 'FOOT_INITIAL_CONTACT' in event[1]]
        first_hs_event = hs_events[0][2]
        
        last_hs_event = hs_events[-1][2]
        raw = self.load_raw_data(raw_path, frame_init=first_hs_event * 100, frame_end=last_hs_event * 100)

        # Load CDG data
        cdg_path = os.path.join(
            data_path, "biomechanics", "COG.csv")
        cdg = pd.read_csv(cdg_path, sep=";", decimal=",", header=0,)

        # Estimate temporal parameters
        try:
            right_strides_temporal_df = self.temporal_parameters(
                right_events, 'RIGHT')
            left_strides_temporal_df = self.temporal_parameters(
                left_events, 'LEFT')
            self.spatiotemporal_data[subject_id + "_" +
                                     record_id]["temporal"]["left"] = left_strides_temporal_df
            self.spatiotemporal_data[subject_id + "_" +
                                     record_id]["temporal"]["right"] = right_strides_temporal_df
        except:
            logger("temporal parameters for " + subject_id +
                   " " + record_id, msgType=LoggerType.FAIL)
            return -1

        # Estimate spatial parameters
        # try:
        right_strides_spatial_df = self.spatial_parameters(
            right_events, 'RIGHT', raw, cdg)
        left_strides_spatial_df = self.spatial_parameters(
            left_events, 'LEFT', raw, cdg)
        self.spatiotemporal_data[subject_id + "_" +
                                 record_id]["spatial"]["left"] = left_strides_spatial_df
        self.spatiotemporal_data[subject_id + "_" +
                                 record_id]["spatial"]["right"] = right_strides_spatial_df
        # except:
        #    logger("spatial parameters for " + subject_id +
        #           " " + record_id, msgType=LoggerType.FAIL)
        #   return -1

        # Find outliers
        self.find_stride_outliers(
            right_strides_spatial_df, subject_id, record_id, 'RIGHT')
        self.find_stride_outliers(
            left_strides_spatial_df, subject_id, record_id, 'LEFT')

        return 0

    def find_outliers(self):
        '''TODO'''
        # Load filtering file
        self.load_stride_filtering()

        # Estimate spatiotemporal parameters
        for subject in self.records.keys():
            logger("[Spatiotemporal Estimator] Process " +
                   subject, msgType=LoggerType.OKCYAN)
            for record_id in self.records[subject]:
                logger("[Spatiotemporal Estimator] Record " +
                       record_id, msgType=LoggerType.OKCYAN)
                result = self.process_record(self.dataset, subject, record_id)
                if result < 0:
                    self.errors.append(subject + "_" + record_id)

        # Update filtering file
        self.stride_filtering.to_excel(self.filtering_path, index=False)

    def filter_spatiotemporal(self, subject, record):
        '''TODO'''
        spatiotemporal = self.spatiotemporal_data[subject + "_" + record]

        # Get filtering for this record
        filtering = self.stride_filtering.loc[(self.stride_filtering['Record'] == record) & (
            self.stride_filtering['Subject'] == subject)]
        left_filtering = ast.literal_eval(
            filtering['Left_Strides_Filtering'].values.tolist()[0])
        left_filtering = list(map(lambda x: x - 1, left_filtering))

        right_filtering = ast.literal_eval(
            filtering['Right_Strides_Filtering'].values.tolist()[0])
        right_filtering = list(map(lambda x: x - 1, right_filtering))

        spatiotemporal['temporal']['left'].drop(
            spatiotemporal['temporal']['left'].index[left_filtering], inplace=True)
        spatiotemporal['temporal']['right'].drop(
            spatiotemporal['temporal']['right'].index[right_filtering], inplace=True)
        spatiotemporal['spatial']['left'].drop(
            spatiotemporal['spatial']['left'].index[left_filtering], inplace=True)
        spatiotemporal['spatial']['right'].drop(
            spatiotemporal['spatial']['right'].index[right_filtering], inplace=True)

    def generate_filtered_resume(self):
        '''TODO'''
        # Reload stride filtering file
        self.load_stride_filtering(add_spatitemporal=False)

        rows = []
        for subject in self.records.keys():
            logger("[Spatiotemporal] Generate filtered resume " +
                   subject, msgType=LoggerType.OKCYAN)
            for record_id in self.records[subject]:
                logger("[Spatiotemporal] Record " + record_id,
                       msgType=LoggerType.OKCYAN)

                # Create dir for filtered data
                data_path = os.path.join(
                    self.dataset, subject, record_id, "preprocessed")
                if not os.path.exists(data_path):
                    os.makedirs(data_path)

                # Remove strides for spatiotemporal data
                self.filter_spatiotemporal(subject, record_id)

                # Generate output
                row = self.write_output(subject, record_id)
                rows.append(row)


        # Write kinematics resume data for all records
        df_spatiotemporal = pd.DataFrame(rows)
        df_spatiotemporal = df_spatiotemporal.round(decimals = 3)
        df_spatiotemporal.to_csv(os.path.join(self.dataset, "spatiotemporal.csv"), index=False)
