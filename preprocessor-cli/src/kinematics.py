#! /usr/bin/env python3
import pandas as pd
import numpy as np
from utils import *
from scipy import stats
import ast


class KinematicsProcessor:
    """Doc"""

    def __init__(self, dataset_path, records, filtering):
        self.dataset = dataset_path
        self.records = records
        self.filtering_path = filtering
        self.stride_filtering = None
        self.kinematics_data = {}

    def load_stride_filtering(self, add_kinematics=True):
        self.stride_filtering = pd.read_excel(self.filtering_path)

        if add_kinematics:
            self.stride_filtering["Kinematics_Left_Strides_Filtering"] = np.empty(
                (len(self.stride_filtering), 0)).tolist()
            self.stride_filtering["Kinematics_Right_Strides_Filtering"] = np.empty(
                (len(self.stride_filtering), 0)).tolist()

    def load_kinematics(self, subject_id, record_id):
        # REVISAR POSIBLE BUG DEL SOFTWARE CLINICAL3DMA:
        # Por ahora, cargamos los datos invertidos. Los nombrados con "left" los consideramos
        # como right y viceversa
        kinematics_path = "D:/dataset/optitrack/" + \
            subject_id + "/" + record_id + "/" + "strides/"
        kinematics = {}

        # ------- Pelvis -------
        # Sagittal Plane - Tilt
        kinematics['Right_Pelvis_Tilt'] = pd.read_csv(
            kinematics_path + "Inclinación pélvica_Izquierda.csv", header=None)
        kinematics['Left_Pelvis_Tilt'] = pd.read_csv(
            kinematics_path + "Inclinación pélvica_Derecha.csv", header=None)

        # Coronal Plane - Obliquity
        kinematics['Right_Pelvis_Obliquity'] = pd.read_csv(
            kinematics_path + "Oblicuidad pélvica_Izquierda.csv", header=None)
        kinematics['Left_Pelvis_Obliquity'] = pd.read_csv(
            kinematics_path + "Oblicuidad pélvica_Derecha.csv", header=None)

        # Transversal Plane - Rotation
        kinematics['Right_Pelvis_Rotation'] = pd.read_csv(
            kinematics_path + "Rotación pélvica_Izquierda.csv", header=None)
        kinematics['Left_Pelvis_Rotation'] = pd.read_csv(
            kinematics_path + "Rotación pélvica_Derecha.csv", header=None)

        # ------- Hip -------
        # Sagittal Plane - Flexion [+]/Extension [-]
        kinematics['Right_Hip_FlexExt'] = pd.read_csv(
            kinematics_path + "Flexión extensión de cadera_Izquierda.csv", header=None)
        kinematics['Left_Hip_FlexExt'] = pd.read_csv(
            kinematics_path + "Flexión extensión de cadera_Derecha.csv", header=None)

        # Coronal Plane - Adduction [+]/Abduction [-]
        kinematics['Right_Hip_AddAbd'] = pd.read_csv(
            kinematics_path + "Abducción aducción de cadera_Izquierda.csv", header=None)
        kinematics['Left_Hip_AddAbd'] = pd.read_csv(
            kinematics_path + "Abducción aducción de cadera_Derecha.csv", header=None)

        # Transversal Plane - Rotation Internal [+] /External [-]
        kinematics['Right_Hip_Rotation'] = pd.read_csv(
            kinematics_path + "Rotación de cadera_Izquierda.csv", header=None)
        kinematics['Left_Hip_Rotation'] = pd.read_csv(
            kinematics_path + "Rotación de cadera_Derecha.csv", header=None)

        # ------- Knee -------
        # Sagittal Plane - Flexion [+]/Extension [-]
        kinematics['Right_Knee_FlexExt'] = pd.read_csv(
            kinematics_path + "Flexión extensión de rodilla_Izquierda.csv", header=None)
        kinematics['Left_Knee_FlexExt'] = pd.read_csv(
            kinematics_path + "Flexión extensión de rodilla_Derecha.csv", header=None)

        # Coronal Plane - Valgus [+]/ Varus [-]
        kinematics['Right_Knee_ValVar'] = pd.read_csv(
            kinematics_path + "Abducción aducción de rodilla_Izquierda.csv", header=None)
        kinematics['Left_Knee_ValVar'] = pd.read_csv(
            kinematics_path + "Abducción aducción de rodilla_Derecha.csv", header=None)

        # Transversal Plane - Rotation Internal [+] /External [-]
        kinematics['Right_Knee_Rotation'] = pd.read_csv(
            kinematics_path + "Rotación de rodilla_Izquierda.csv", header=None)
        kinematics['Left_Knee_Rotation'] = pd.read_csv(
            kinematics_path + "Rotación de rodilla_Derecha.csv", header=None)

        # ------- Ankle -------
        # Sagittal Plane - Flexion [+]/Extension [-]
        kinematics['Right_Ankle_FlexExt'] = pd.read_csv(
            kinematics_path + "Flexión extensión de tobillo_Izquierda.csv", header=None)
        kinematics['Left_Ankle_FlexExt'] = pd.read_csv(
            kinematics_path + "Flexión extensión de tobillo_Derecha.csv", header=None)

        # Coronal Plane - Adduction [+]/Abduction [-]
        kinematics['Right_Ankle_AddAbd'] = pd.read_csv(
            kinematics_path + "Abducción aducción de tobillo_Izquierda.csv", header=None)
        kinematics['Left_Ankle_AddAbd'] = pd.read_csv(
            kinematics_path + "Abducción aducción de tobillo_Derecha.csv", header=None)

        # Transversal Plane - Rotation Internal [+] /External [-]
        kinematics['Right_Ankle_Rotation'] = pd.read_csv(
            kinematics_path + "Rotación de tobillo_Izquierda.csv", header=None)
        kinematics['Left_Ankle_Rotation'] = pd.read_csv(
            kinematics_path + "Rotación de tobillo_Derecha.csv", header=None)

        # ------- Foot -------
        # Transversal Plane - Rotation Internal [+] /External [-]
        kinematics['Right_Foot_Rotation'] = pd.read_csv(
            kinematics_path + "Rotación de pie_Izquierda.csv", header=None)
        kinematics['Left_Foot_Rotation'] = pd.read_csv(
            kinematics_path + "Rotación de pie_Derecha.csv", header=None)

        self.kinematics_data[subject_id + "_" + record_id] = kinematics

        return kinematics

    def find_stride_outliers(self, kinematics, subject, record, foot):
        threshold_z = 2
        threshold_std = 100
        threshold_angle = -60

        outliers = np.unique(np.where(kinematics.min() < threshold_angle)[0]).tolist()
        std = kinematics.min().std()
        if len(outliers) == 0 and std > threshold_std:
            z = np.abs(stats.zscore(kinematics.min()))
            outliers = list(set(outliers + np.unique(np.where(z > threshold_z)[0]).tolist()))

        # Add to table
        index = self.stride_filtering.loc[(self.stride_filtering["Subject"] == subject)
                                          & (self.stride_filtering["Record"] == record)].index[0]

        self.stride_filtering.at[index, 'Kinematics_' +
                                 (foot.title()) + '_Strides_Filtering'] = outliers

    def process_record(self, dataset, subject_id, record_id):
        data_path = os.path.join(dataset, subject_id, record_id)
        kinematics = self.load_kinematics(subject_id, record_id)

        # Find outliers
        self.find_stride_outliers(
            kinematics['Right_Pelvis_Tilt'], subject_id, record_id, 'RIGHT')
        self.find_stride_outliers(
            kinematics['Left_Pelvis_Tilt'], subject_id, record_id, 'LEFT')

        return 0

    def run(self):
        # Load filtering file
        self.load_stride_filtering()

        for subject in self.records.keys():
            logger("[Kinematics] Process " + subject,
                   msgType=LoggerType.OKBLUE)
            for record_id in self.records[subject]:
                logger("[Kinematics] Record " + record_id,
                       msgType=LoggerType.OKBLUE)
                result = self.process_record(self.dataset, subject, record_id)
                if result < 0:
                    self.errors.append(subject + "_" + record_id)

        # Update filtering file
        self.stride_filtering.to_excel(self.filtering_path, index=False)

    def filter_kinematics(self, subject, record):
        kinematics = self.kinematics_data[subject + "_" + record]

        # Get filtering for this record
        filtering = self.stride_filtering.loc[(self.stride_filtering['Record'] == record) & (
            self.stride_filtering['Subject'] == subject)]
        left_filtering = ast.literal_eval(
            filtering['Left_Strides_Filtering'].values.tolist()[0])
        right_filtering = ast.literal_eval(
            filtering['Right_Strides_Filtering'].values.tolist()[0])

        for key in kinematics:
            foot = key.split("_")[0].lower()
            data = self.kinematics_data[subject + "_" + record][key]
            if foot == "left":
                self.kinematics_data[subject + "_" + record][key].drop(data.columns[left_filtering], axis=1, inplace=True)
            elif foot == "right":
                self.kinematics_data[subject + "_" + record][key].drop(data.columns[right_filtering], axis=1, inplace=True)

            # transpose dataframe: rows are strides
            #self.kinematics_data[subject + "_" + record][key] = self.kinematics_data[subject + "_" + record][key].T

    def generate_filtered_resume(self):
        # Reload stride filtering file
        self.load_stride_filtering(add_kinematics=False)
        for subject in self.records.keys():
            logger("[Kinematics] Generate filtered resume " +
                   subject, msgType=LoggerType.OKBLUE)
            for record_id in self.records[subject]:
                logger("[Kinematics] Record " + record_id,
                       msgType=LoggerType.OKBLUE)
                
                # Remove strides for kinematics data
                self.filter_kinematics(subject, record_id)

                # Generate kinematics summary
                current_kinematics_data = self.kinematics_data[subject + "_" + record_id]
                kinematics_resume_list = []
                for key in current_kinematics_data.keys():
                    foot = key.split("_")[0]
                    kinematics_resume = {"Parameter": key}
                    kinematics_resume["Min"] = current_kinematics_data[key].to_numpy(
                    ).min()
                    kinematics_resume["Max"] = current_kinematics_data[key].to_numpy(
                    ).max()
                    kinematics_resume["Range"] = abs(
                        kinematics_resume["Max"] - kinematics_resume["Min"])
                    kinematics_resume["Mean"] = current_kinematics_data[key].to_numpy(
                    ).mean()
                    kinematics_resume["Stride_Start"] = current_kinematics_data[key].loc[0].mean(
                    )
                    kinematics_resume["Stride_End"] = current_kinematics_data[key].loc[99].mean(
                    )

                    # Load spatiotemporal
                    spatiotemporal = pd.read_excel(self.dataset + "/" + subject + "/" 
                                                    + record_id + "/preprocessed/" 
                                                    + subject 
                                                    + "_" + record_id + ".spatiotemporal.xlsx", 
                                sheet_name=(foot + '_Gait_Cycle'), index_col=0)
                    transitions = spatiotemporal.stance_percent.round().values.tolist()
                    kinematics_resume["Phase_Transition"] = np.mean([current_kinematics_data[key].loc[percent].values[idx] for idx,percent in enumerate(transitions)])
                    kinematics_resume_list.append(kinematics_resume)

                # Create a Pandas Excel writer using XlsxWriter as the engine.
                output_path = os.path.join(self.dataset, subject, record_id, "preprocessed", (
                    subject + "_" + record_id + ".kinematics.xlsx"))

                # Write kinematics resume data
                df_kinematics_resume = pd.DataFrame(kinematics_resume_list)
                df_kinematics_resume.to_excel(output_path, index=False)
