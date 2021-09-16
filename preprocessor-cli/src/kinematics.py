#! /usr/bin/env python3
'''TODO'''
import ast
import pandas as pd
import numpy as np
from utils import *
from scipy import stats
from scipy.signal import savgol_filter
import os

class KinematicsProcessor:
    '''TODO'''

    def __init__(self, dataset_path, records, filtering, filtering_threshold, enable_logger=True):
        '''TODO'''
        self.dataset = dataset_path
        self.records = records
        self.filtering_path = filtering
        self.stride_filtering = None
        self.enable_logger= enable_logger
        self.kinematics_data = {}
        self.kinematics_filtering_threshold = filtering_threshold

    def log(self, text, msgType=LoggerType.OKGREEN):
        if self.enable_logger:
            logger(text, msgType=msgType)

    def load_stride_filtering(self, add_kinematics=True):
        '''TODO'''
        self.stride_filtering = pd.read_excel(self.filtering_path)

        if add_kinematics:
            self.stride_filtering["Kinematics_Left_Strides_Filtering"] = np.empty(
                (len(self.stride_filtering), 0)).tolist()
            self.stride_filtering["Kinematics_Right_Strides_Filtering"] = np.empty(
                (len(self.stride_filtering), 0)).tolist()

    def load_kinematics(self, subject_id, record_id):
        '''TODO'''
        kinematics_path = os.path.join(self.dataset, subject_id, record_id, "strides/")
        kinematics = {}

        # ------- Pelvis -------
        # Sagittal Plane - Tilt
        kinematics['right_pelvis_tilt'] = pd.read_csv(kinematics_path + "Pelvic_Tilt_Right.csv", header=None)
        kinematics['left_pelvis_tilt'] = pd.read_csv(kinematics_path + "Pelvic_Tilt_Left.csv", header=None)
        
        
        # Coronal Plane - Obliquity
        kinematics['right_pelvis_obliquity'] = pd.read_csv(kinematics_path + "Pelvic_Obliquity_Right.csv", header=None)
        kinematics['left_pelvis_obliquity'] = pd.read_csv(kinematics_path + "Pelvic_Obliquity_Left.csv", header=None)
        
        # Transversal Plane - Rotation
        kinematics['right_pelvis_rotation'] = pd.read_csv(kinematics_path + "Pelvic_Rotation_Right.csv", header=None)
        kinematics['left_pelvis_rotation'] = pd.read_csv(kinematics_path + "Pelvic_Rotation_Left.csv", header=None)

        # ------- Hip -------
        # Sagittal Plane - Flexion [+]/Extension [-]
        kinematics['right_hip_flex_ext'] = pd.read_csv(kinematics_path + "Hip_FlexExt_Right.csv", header=None)
        kinematics['left_hip_flex_ext'] = pd.read_csv(kinematics_path + "Hip_FlexExt_Left.csv", header=None)
        
        # Coronal Plane - Adduction [+]/Abduction [-]
        kinematics['right_hip_add_abd'] = pd.read_csv(kinematics_path + "Hip_AbdAdd_Right.csv", header=None) 
        kinematics['left_hip_add_abd'] = pd.read_csv(kinematics_path + "Hip_AbdAdd_Left.csv", header=None)

        # Transversal Plane - Rotation Internal [+] /External [-]
        kinematics['right_hip_rotation'] = pd.read_csv(kinematics_path + "Hip_Rotation_Right.csv", header=None)
        kinematics['left_hip_rotation'] = pd.read_csv(kinematics_path + "Hip_Rotation_Left.csv", header=None)
        
        # ------- Knee -------
        # Sagittal Plane - Flexion [+]/Extension [-]
        kinematics['right_knee_flex_ext'] = pd.read_csv(kinematics_path + "Knee_FlexExt_Right.csv", header=None)
        kinematics['left_knee_flex_ext'] = pd.read_csv(kinematics_path + "Knee_FlexExt_Left.csv", header=None)

        # Coronal Plane - Valgus [+]/ Varus [-]
        kinematics['right_knee_val_var'] = pd.read_csv(kinematics_path + "Knee_AbdAdd_Right.csv", header=None)
        kinematics['left_knee_val_var'] = pd.read_csv(kinematics_path + "Knee_AbdAdd_Left.csv", header=None)

        # Transversal Plane - Rotation Internal [+] /External [-]
        kinematics['right_knee_rotation'] =pd.read_csv(kinematics_path + "Knee_Rotation_Right.csv", header=None)
        kinematics['left_knee_rotation'] = pd.read_csv(kinematics_path + "Knee_Rotation_Left.csv", header=None)
        
        # ------- Ankle -------
        # Sagittal Plane - Flexion [+]/Extension [-]
        kinematics['right_ankle_flex_ext'] =pd.read_csv(kinematics_path + "Ankle_FlexExt_Right.csv", header=None)
        kinematics['left_ankle_flex_ext'] = pd.read_csv(kinematics_path + "Ankle_FlexExt_Left.csv", header=None)

        # Coronal Plane - Adduction [+]/Abduction [-]
        kinematics['right_ankle_add_abd'] = pd.read_csv(kinematics_path + "Ankle_AbdAdd_Right.csv", header=None)
        kinematics['left_ankle_add_abd'] = pd.read_csv(kinematics_path + "Ankle_AbdAdd_Left.csv", header=None)
        
        # Transversal Plane - Rotation Internal [+] /External [-]
        kinematics['right_ankle_rotation'] = pd.read_csv(kinematics_path + "Ankle_Rotation_Right.csv", header=None)
        kinematics['left_ankle_rotation'] = pd.read_csv(kinematics_path + "Ankle_Rotation_Left.csv", header=None)
        
        # ------- Foot -------
        #Transversal Plane - Rotation Internal [+] /External [-]
        kinematics['right_foot_rotation'] = pd.read_csv(kinematics_path + "Foot_Rotation_Right.csv", header=None)
        kinematics['left_foot_rotation'] = pd.read_csv(kinematics_path + "Foot_Rotation_Left.csv", header=None)

        # Apply filtering for noise reduction and Reindex columns to start at 1 rather than 0
        for key in kinematics:
            kinematics[key] = kinematics[key].apply(lambda col: apply_savgol_filter(col))
            kinematics[key].columns += 1

        self.kinematics_data[subject_id + "_" + record_id] = kinematics
        return kinematics
    ''' 
    def find_stride_outliers(self, kinematics, subject, record, foot):
        threshold_z = 2
        threshold_std = 100
        threshold_angle = -60

        outliers = (np.unique(np.where(kinematics.min() < threshold_angle)[0]) + 1).tolist()
        std = kinematics.min().std()
        if len(outliers) == 0 and std > threshold_std:
            z = np.abs(stats.zscore(kinematics.min()))
            outliers = list(
                set(outliers + np.unique(np.where(z > threshold_z)[0]).tolist()))

        # Add to table
        index = self.stride_filtering.loc[(self.stride_filtering["Subject"] == subject)
                                          & (self.stride_filtering["Record"] == record)].index[0]

        self.stride_filtering.at[index, 'Kinematics_' +
                                 (foot.title()) + '_Strides_Filtering'] = outliers
    '''

    def find_stride_outliers(self, kinematics, subject, record, foot):
        outliers = []
        for key in kinematics:
            angle = ("_").join(key.split("_")[1:])
            if not angle in self.kinematics_filtering_threshold:
                continue
                
            min_threshold = self.kinematics_filtering_threshold[angle]["min"]
            max_threshold = self.kinematics_filtering_threshold[angle]["max"]
            new_outliers =  np.unique(np.where((kinematics[key].min() < min_threshold) 
                                            | (kinematics[key].min() > max_threshold))[0]).tolist()
            outliers = list(set(outliers + new_outliers))

        # Add to table
        index = self.stride_filtering.loc[(self.stride_filtering["Subject"] == subject)
                                          & (self.stride_filtering["Record"] == record)].index[0]

        self.stride_filtering.at[index, 'Kinematics_' +
                                        (foot.title()) + '_Strides_Filtering'] = outliers

    def process_record(self, dataset, subject_id, record_id):
        '''TODO'''
        data_path = os.path.join(dataset, subject_id, record_id)
        kinematics = self.load_kinematics(subject_id, record_id)

        # Find outliers
        self.find_stride_outliers(kinematics, subject_id, record_id, 'RIGHT')
        self.find_stride_outliers(kinematics, subject_id, record_id, 'LEFT')

        return 0

    def run(self):
        '''TODO'''
        # Load filtering file
        self.load_stride_filtering()

        for subject in self.records.keys():
            self.log("[Kinematics] Process " + subject,
                   msgType=LoggerType.OKBLUE)
            for record_id in self.records[subject]:
                self.log("[Kinematics] Record " + record_id,
                       msgType=LoggerType.OKBLUE)
                result = self.process_record(self.dataset, subject, record_id)
                if result < 0:
                    self.errors.append(subject + "_" + record_id)

        # Update filtering file
        self.stride_filtering.to_excel(self.filtering_path, index=False)

    def filter_kinematics(self, subject, record):
        '''TODO'''
        kinematics = self.kinematics_data[subject + "_" + record]

        # Get filtering for this record
        filtering = self.stride_filtering.loc[(self.stride_filtering['Record'] == record) & (
            self.stride_filtering['Subject'] == subject)]
        left_filtering = ast.literal_eval(
            filtering['Left_Strides_Filtering'].values.tolist()[0])
        left_filtering = list(map(lambda x: x - 1, left_filtering))

        right_filtering = ast.literal_eval(
            filtering['Right_Strides_Filtering'].values.tolist()[0])
        right_filtering = list(map(lambda x: x - 1, right_filtering))
        

        for key in kinematics:
            
            foot = key.split("_")[0].lower()
            data = self.kinematics_data[subject + "_" + record][key]
            if foot == "left":
                self.kinematics_data[subject + "_" + record][key].drop(
                    data.columns[left_filtering], axis=1, inplace=True)
            elif foot == "right":
                self.kinematics_data[subject + "_" + record][key].drop(
                    data.columns[right_filtering], axis=1, inplace=True)

            # transpose dataframe: rows are strides
            #self.kinematics_data[subject + "_" + record][key] = self.kinematics_data[subject + "_" + record][key].T

    def generate_filtered_resume(self):
        '''TODO'''
        # Reload stride filtering file
        self.load_stride_filtering(add_kinematics=False)
        rows = []
        for subject in self.records.keys():
            self.log("[Kinematics] Generate filtered resume " +
                   subject, msgType=LoggerType.OKBLUE)
            for record_id in self.records[subject]:
                self.log("[Kinematics] Record " + record_id,
                       msgType=LoggerType.OKBLUE)
                    
                
                row = {'subject': subject, 'record': record_id}

                # Remove strides for kinematics data
                self.filter_kinematics(subject, record_id)

                # Load spatiotemporal resume
                spatiotemporal_left = pd.read_excel(self.dataset + "/" + subject + "/" 
                    + record_id + "/preprocessed/"
                    + subject
                    + "_" + record_id + ".spatiotemporal.xlsx",
                    sheet_name=('Left_Gait_Cycle'), index_col=0)

                spatiotemporal_right = pd.read_excel(self.dataset + "/" + subject + "/" 
                    + record_id + "/preprocessed/"
                    + subject
                    + "_" + record_id + ".spatiotemporal.xlsx",
                    sheet_name=('Right_Gait_Cycle'), index_col=0)

                transitions_left = spatiotemporal_left.stance_percent.round().values.tolist()
                transitions_right = spatiotemporal_right.stance_percent.round().values.tolist()

                # Generate kinematics summary
                current_kinematics_data = self.kinematics_data[subject + "_" + record_id]
                kinematics_resume_list = []
                for key in current_kinematics_data.keys():
                    foot = key.split("_")[0]
                    transitions = transitions_left if foot.lower() == 'left' else transitions_right
                    kinematics_resume = {"Parameter": key}
                    kinematics_resume["Min"] = current_kinematics_data[key].min().mean()
                    kinematics_resume["Max"] = current_kinematics_data[key].max().mean()
                    kinematics_resume["Range"] = abs(
                        kinematics_resume["Max"] - kinematics_resume["Min"])
                    kinematics_resume["Mean"] = current_kinematics_data[key].to_numpy(
                    ).mean()
                    kinematics_resume["Stride_Start"] = current_kinematics_data[key].loc[0].mean(
                    )
                    kinematics_resume["Stride_End"] = current_kinematics_data[key].loc[99].mean(
                    )

                    kinematics_resume["Swing_Angle"] = np.mean(
                        [current_kinematics_data[key].loc[percent].values[idx] for idx, percent in enumerate(transitions)])
                    kinematics_resume_list.append(kinematics_resume)

                    for item in kinematics_resume.keys():
                        if item not in ['Parameter']:
                            row[key.lower() + "_" + item] = kinematics_resume[item]
                
                rows.append(row)

                # Create a Pandas Excel writer using XlsxWriter as the engine.
                output_path = os.path.join(self.dataset, subject, record_id, "preprocessed", (
                    subject + "_" + record_id + ".kinematics.xlsx"))

                # Write kinematics resume data for this record
                df_kinematics_resume = pd.DataFrame(kinematics_resume_list)
                df_kinematics_resume.to_excel(output_path, index=False)

        # Write kinematics resume data for all records
        df_kinematics = pd.DataFrame(rows)
        df_kinematics = df_kinematics.round(decimals = 3)
        df_kinematics.to_csv(os.path.join(self.dataset, "kinematics.csv"), index=False)
