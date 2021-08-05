#! /usr/bin/env python3
import pandas as pd
import numpy as np
from utils import *
import ast


class SpatioTemporalEstimator:
    """Doc"""

    def __init__(self, dataset_path, records, filtering):
        self.dataset = dataset_path
        self.records = records
        self.filtering_path = filtering
        self.turn_list = []
        self.errors = []
        self.stride_filtering = None
        self.spatiotemporal_data = {}

    def load_stride_filtering(self, add_spatitemporal=True):
        self.stride_filtering = pd.read_excel(self.filtering_path)

        if add_spatitemporal:
            self.stride_filtering["Spatiotemporal_Left_Strides_Filtering"] = np.empty(
                (len(self.stride_filtering), 0)).tolist()
            self.stride_filtering["Spatiotemporal_Right_Strides_Filtering"] = np.empty(
                (len(self.stride_filtering), 0)).tolist()

    def load_raw_data(self, path):
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

        raw = pd.read_csv(path, sep='\t', lineterminator='\n', header=None, skiprows=[0],
                          usecols=range(len(headers)), names=headers)

        return raw

    def sride_length(self, stride, foot, raw):
        start = stride[0][2]  # Foot Initial Contact 1
        end = stride[len(stride) - 1][2]  # Foot Initial Contact 2

        if not np.isnan(start) and not np.isnan(end):
            pos_init = raw[foot + "_heel_PZ"][int(start * 100)]
            pos_end = raw[foot + "_heel_PZ"][int(end * 100)]
            return abs(pos_end - pos_init)
        else:
            return np.NaN

    def step_length(self, stride, foot, raw):
        start = getEventTime(stride, 'EVENT_' + opposite_foot(foot.upper()) +
                             '_FOOT_INITIAL_CONTACT')  # Opposite Foot Initial Contact
        end = stride[len(stride) - 1][2]  # Foot Initial Contact 2

        if not np.isnan(start) and not np.isnan(end):
            pos_init = raw[foot + "_heel_PZ"][int(start * 100)]
            pos_end = raw[opposite_foot(
                foot).lower() + "_heel_PZ"][int(end * 100)]
            return abs(pos_end - pos_init)
        else:
            return np.NaN

    def stride_heel_height(self, stride, foot, raw):
        start = stride[0][2]  # Foot Initial Contact 1
        end = stride[len(stride) - 1][2]  # Foot Initial Contact 2

        if not np.isnan(start) and not np.isnan(end):
            return (raw[foot + "_heel_PY"][int(start * 100):int((end * 100) + 10)].max())
        else:
            return np.NaN

    def base_of_support(self, stride, raw):
        start = stride[0][2]  # Foot Initial Contact 1
        end = stride[len(stride) - 1][2]  # Foot Initial Contact 2

        if not np.isnan(start) and not np.isnan(end):
            pos_left = raw["left_heel_PX"][int(
                start * 100): int(end * 100)].values
            pos_right = raw["right_heel_PX"][int(
                start * 100): int(end * 100)].values
            return abs(pos_left - pos_right).mean()
        else:
            return np.NaN

    def spatial_parameters(self, stride_events, foot, raw):
        foot = foot.lower()
        op_foot = opposite_foot(foot).lower()

        # Prepare dataframe
        df = pd.DataFrame([0 for stride in stride_events],
                          columns=["stride_length", ])

        df.index.name = 'stride'

        # initial contact 1 (stride[0][2]) y initial contact 2 (stride[4][2])
        df["stride_length"] = list(
            map(lambda stride: self.sride_length(stride, foot, raw), stride_events))
        df["step_length"] = list(
            map(lambda stride: self.step_length(stride, foot, raw), stride_events))
        df["max_heel_height"] = list(
            map(lambda stride: self.stride_heel_height(stride, foot, raw), stride_events))
        df["base_of_support"] = list(
            map(lambda stride: self.base_of_support(stride, raw), stride_events))

        # Start row index from 1
        # df.index = df.index + 1

        df.round(2)
        return df

    def temporal_parameters(self, stride_events, foot):
        foot = foot.lower()
        op_foot = opposite_foot(foot).lower()

        # Prepare dataframe
        df = pd.DataFrame([stride[0][2] for stride in stride_events], columns=[
                          (foot + '_initial_contact_1')])
        df[(op_foot + '_toe_off')] = list(map(lambda stride: getEventTime(stride,
                                                                          'EVENT_' + op_foot.upper() + '_FOOT_TOE_OFF'), stride_events))
        df[(op_foot + '_initial_contact')] = list(map(lambda stride: getEventTime(stride,
                                                                                  'EVENT_' + op_foot.upper() + '_FOOT_INITIAL_CONTACT'), stride_events))
        df[(foot + '_toe_off')] = list(map(lambda stride: getEventTime(stride,
                                                                       'EVENT_' + foot.upper() + '_FOOT_TOE_OFF'), stride_events))
        df[(foot + '_initial_contact_2')] = [stride[len(stride) - 1][2]
                                             for stride in stride_events]

        df.index.name = 'stride'

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
        df['double_support1_duration'] = df[(
            op_foot + '_toe_off')] - df[(foot + '_initial_contact_1')]
        df['double_support2_duration'] = df[(
            foot + '_toe_off')] - df[(op_foot + '_initial_contact')]
        df['double_support_percent'] = ((df['double_support1_duration'] +
                                        df['double_support2_duration']) / df['stride_duration']) * 100

        # Start row index from 1
        # df.index = df.index + 1

        df.round(2)
        return df

    def estimate_gait_time(self, left_strides_temporal_df, right_strides_temporal_df):
        gait_start = min([left_strides_temporal_df.left_initial_contact_1.iloc[0],
                         right_strides_temporal_df.right_initial_contact_1.iloc[0]])

        gait_end = max([left_strides_temporal_df.left_initial_contact_2.max(
        ), right_strides_temporal_df.right_initial_contact_2.max()])

        return gait_end - gait_start

    def estimate_cadence(self, gait_time, total_strides_left, total_strides_right):
        steps = (min([total_strides_left, total_strides_right]) * 2) + \
            (abs(total_strides_left - total_strides_right) * 2)
        return round((steps * 60) / gait_time, 2)

    def write_output(self, subject_id, record_id):
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

        overview = {
            'Total_Strides_Left': len(left_strides_temporal),
            'Total_Strides_Right': len(right_strides_temporal),
            'Gait_Time': gait_time,
            'Cadence': self.estimate_cadence(gait_time, len(left_strides_temporal), len(right_strides_temporal)),
            # 'Speed': 0,
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
            # 'Double_Support_Percent': (left_means['ds_percent'] + right_means['ds_percent']) / 2,
            'Right_Stride_Length': right_means['stride_length'],
            'Left_Stride_Length': left_means['stride_length'],
            'Right_Step_Length': right_means['step_length'],
            'Left_Step_Length': left_means['step_length'],
            'Base_Of_Support': ((left_means['base_of_support'] + right_means['base_of_support']) / 2).round(2),
            'Right_Heel_Height': right_means['max_heel_height'],
            'Left_Heel_Height': left_means['max_heel_height'],
        }

        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(output_path, engine='openpyxl')

        # Write workseet from overview map
        write_sheet_from_dict(writer, 'Overview', overview)

        # Write each dataframe to a different worksheet.
        write_sheet_from_df(writer, 'Right_Gait_Cycle',
                            right_strides_temporal)
        write_sheet_from_df(writer, 'Left_Gait_Cycle', left_strides_temporal)

        write_sheet_from_df(writer, 'Right_Spatial',
                            right_strides_spatial)
        write_sheet_from_df(writer, 'Left_Spatial', left_strides_spatial)

        # Close the Pandas Excel writer and output the Excel file.
        writer.save()

    def find_stride_outliers(self, strides_spatial_df, subject, record, foot):
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

        # Add to table
        index = self.stride_filtering.loc[(self.stride_filtering["Subject"] == subject)
                                          & (self.stride_filtering["Record"] == record)].index[0]

        self.stride_filtering.at[index, 'Spatiotemporal_' + (
            foot.title()) + '_Strides_Filtering'] = list(set(nan_values + outliers))

    def process_record(self, dataset, subject_id, record_id):
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
        raw = self.load_raw_data(raw_path)

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
        try:
            right_strides_spatial_df = self.spatial_parameters(
                right_events, 'RIGHT', raw)
            left_strides_spatial_df = self.spatial_parameters(
                left_events, 'LEFT', raw)
            self.spatiotemporal_data[subject_id + "_" +
                                     record_id]["spatial"]["left"] = left_strides_spatial_df
            self.spatiotemporal_data[subject_id + "_" +
                                     record_id]["spatial"]["right"] = right_strides_spatial_df
        except:
            logger("spatial parameters for " + subject_id +
                   " " + record_id, msgType=LoggerType.FAIL)
            return -1

        # Find outliers
        self.find_stride_outliers(
            right_strides_spatial_df, subject_id, record_id, 'RIGHT')
        self.find_stride_outliers(
            left_strides_spatial_df, subject_id, record_id, 'LEFT')

        return 0

    def find_outliers(self):
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
        spatiotemporal = self.spatiotemporal_data[subject + "_" + record]

        # Get filtering for this record
        filtering = self.stride_filtering.loc[(self.stride_filtering['Record'] == record) & (
            self.stride_filtering['Subject'] == subject)]
        left_filtering = ast.literal_eval(
            filtering['Left_Strides_Filtering'].values.tolist()[0])
        right_filtering = ast.literal_eval(
            filtering['Right_Strides_Filtering'].values.tolist()[0])

        spatiotemporal['temporal']['left'].drop(spatiotemporal['temporal']['left'].index[left_filtering], inplace=True)
        spatiotemporal['temporal']['right'].drop(spatiotemporal['temporal']['right'].index[right_filtering], inplace=True)
        spatiotemporal['spatial']['left'].drop(spatiotemporal['spatial']['left'].index[left_filtering], inplace=True)
        spatiotemporal['spatial']['right'].drop(spatiotemporal['spatial']['right'].index[right_filtering], inplace=True)

    def generate_filtered_resume(self):
        # Reload stride filtering file
        self.load_stride_filtering(add_spatitemporal=False)

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
                self.write_output(subject, record_id)
