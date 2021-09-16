import pandas as pd
from docx.shared import Cm
from docxtpl import DocxTemplate, InlineImage
from docx.shared import Cm, Inches, Mm, Emu
import matplotlib.pyplot as plt
import datetime
import win32com.client
import os
import numpy as np
import pathlib

import sys
sys.path.append('../preprocessor-cli/src')
from utils import apply_savgol_filter

BIGGER_SIZE = 18
MEDIUM_SIZE = 14
SMALL_SIZE = 12
LEFT_COLOR = "tab:red"  # 214. 39. 40. (RGB)    #d62727
RIGHT_COLOR = "tab:blue"  # 21. 119. 180. (RGB) #1577b4

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


class ReportGenerator:
    '''TODO'''

    def __init__(self, data_path, general_info, poma_preds, template_path, output_path):
        self.data_path = data_path
        self.subject = "subject"
        self.record = "rec"
        self.preprocessed_path = os.path.join(
            self.data_path, self.subject, self.record, "preprocessed/")
        self.general_info = general_info
        self.poma_preds = poma_preds
        self.template = DocxTemplate(template_path)
        self.output_path = output_path


    def save_fig(self, fig_id, out_dir, tight_layout=True, fig_extension="png", resolution=1200):
        path = os.path.join(out_dir, fig_id + "." + fig_extension)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution)

    def load_data(self):
        self.kinematics_resume = pd.read_excel(
            self.preprocessed_path + (self.subject + "_" + self.record + ".kinematics.xlsx"))

        spatiotemporal_file = self.subject + "_" + self.record + ".spatiotemporal.xlsx"
        spatiotemporal_ovw = pd.read_excel(
            self.preprocessed_path + spatiotemporal_file, header=None)

        spatiotemporal_ovw = spatiotemporal_ovw.T
        spatiotemporal_ovw.columns = spatiotemporal_ovw.iloc[0]
        spatiotemporal_ovw = spatiotemporal_ovw[1:]
        self.spatiotemporal_ovw = spatiotemporal_ovw

        self.spatiotemporal_right_cicle = pd.read_excel(self.preprocessed_path + spatiotemporal_file,
                                                        sheet_name="Right_Gait_Cycle")

        self.spatiotemporal_left_cicle = pd.read_excel(self.preprocessed_path + spatiotemporal_file,
                                                       sheet_name="Left_Gait_Cycle")
        self.spatiotemporal_right_spatial = pd.read_excel(self.preprocessed_path + spatiotemporal_file,
                                                          sheet_name="Right_Spatial")
        self.spatiotemporal_left_spatial = pd.read_excel(self.preprocessed_path + spatiotemporal_file,
                                                         sheet_name="Left_Spatial")

        self.spatiotemporal_left_spatial["stride_length"] = self.spatiotemporal_left_spatial["stride_length"] / 1000
        self.spatiotemporal_left_spatial["step_length"] = self.spatiotemporal_left_spatial["step_length"] / 1000
        self.spatiotemporal_left_spatial["max_heel_height"] = self.spatiotemporal_left_spatial["max_heel_height"] / 1000
        self.spatiotemporal_left_spatial["max_toe_height"] = self.spatiotemporal_left_spatial["max_toe_height"] / 1000
        self.spatiotemporal_left_spatial["base_of_support"] = self.spatiotemporal_left_spatial["base_of_support"] / 1000
        self.spatiotemporal_right_spatial["stride_length"] = self.spatiotemporal_right_spatial["stride_length"] / 1000
        self.spatiotemporal_right_spatial["step_length"] = self.spatiotemporal_right_spatial["step_length"] / 1000
        self.spatiotemporal_right_spatial["max_heel_height"] = self.spatiotemporal_right_spatial["max_heel_height"] / 1000
        self.spatiotemporal_right_spatial["max_toe_height"] = self.spatiotemporal_right_spatial["max_toe_height"] / 1000
        self.spatiotemporal_right_spatial["base_of_support"] = self.spatiotemporal_right_spatial["base_of_support"] / 1000

        kinematics_path = self.preprocessed_path = os.path.join(
            self.data_path, self.subject, self.record, "strides/")
        self.kinematics = self.load_kinematics(kinematics_path)

    def load_kinematics(self, kinematics_path):
        '''TODO'''
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
            kinematics[key] = kinematics[key].apply(
                lambda col: apply_savgol_filter(col))
            kinematics[key].columns += 1

        self.kinematics = kinematics
        return kinematics

    def standard_plot(self, kinematic_name, title, xlabel="Ciclo (%)", y_label="Ángulo (º)"):
        sd_opacity = 0.15
        gc_opacity = 0.035

        plt.axhline(color='black', linewidth=0.5)
        plt.axvspan(0, 60, ymin=0.0, ymax=1, alpha=gc_opacity, color='black')

        # Prepare data
        left_mean = self.kinematics["left_" +
                                    kinematic_name].mean(axis=1).values
        left_mean_plus_sd = left_mean + \
            self.kinematics["left_" + kinematic_name].std(axis=1).values
        left_mean_minus_sd = left_mean - \
            self.kinematics["left_" + kinematic_name].std(axis=1).values

        right_mean = self.kinematics["right_" +
                                     kinematic_name].mean(axis=1).values
        right_mean_plus_sd = right_mean + \
            self.kinematics["right_" + kinematic_name].std(axis=1).values
        right_mean_minus_sd = right_mean - \
            self.kinematics["right_" + kinematic_name].std(axis=1).values

        # Plot data
        graph = plt.plot(left_mean, LEFT_COLOR, right_mean, RIGHT_COLOR,)
        plt.fill_between(range(100), left_mean_plus_sd,
                         left_mean_minus_sd, color=LEFT_COLOR, alpha=sd_opacity)
        plt.fill_between(range(100), right_mean_plus_sd,
                         right_mean_minus_sd, color=RIGHT_COLOR, alpha=sd_opacity)

        # Set title and labels
        plt.title(title, fontsize=BIGGER_SIZE)
        plt.xlabel(xlabel, )
        plt.ylabel(y_label,)
        plt.xticks(np.arange(0, 101, step=20))
        plt.autoscale(enable=True, axis='both', tight=True)
        plt.margins(y=.1)

        return graph

    def plot_kinematics(self, image_name):
        fig = plt.figure(figsize=(15, 19))
        fig.tight_layout()
        plt.subplots_adjust(hspace=0.5, wspace=0.2)

        strides_left = self.kinematics["left_pelvis_tilt"].columns.size
        strides_right = self.kinematics["right_pelvis_tilt"].columns.size

        # Pelvis - Inclinación
        plt.subplot(5, 3, 1)  # Layout 4 rows, 3 columns, position 1 - top left
        self.standard_plot('pelvis_tilt', "Pelvis - Inclinación")

        # Pelvis - Oblicuidad
        plt.subplot(5, 3, 2)
        self.standard_plot('pelvis_obliquity', "Pelvis - Oblicuidad")

        # Pelvis - Rotación
        plt.subplot(5, 3, 3)
        self.standard_plot('pelvis_rotation', "Pelvis - Rotación")

        # Caderas - FlexoExtensión
        plt.subplot(5, 3, 4)
        self.standard_plot('hip_flex_ext', "Cadera - FlexoExtensión")

        # Caderas - Abducción
        plt.subplot(5, 3, 5)
        self.standard_plot('hip_add_abd', "Cadera - Abducción")

        # Caderas - Rotación
        plt.subplot(5, 3, 6)
        self.standard_plot('hip_rotation', "Cadera - Rotación")

        # Rodillas - FlexoExtensión
        plt.subplot(5, 3, 7)
        self.standard_plot('knee_flex_ext', "Rodilla - FlexoExtensión")

        # Rodillas - Abducción
        plt.subplot(5, 3, 8)
        self.standard_plot('knee_val_var', "Rodilla - Abducción")

        # Rodillas - Rotación
        plt.subplot(5, 3, 9)
        self.standard_plot('knee_rotation', "Rodilla - Rotación")

        # Tobillos - FlexoExtensión
        plt.subplot(5, 3, 10)
        self.standard_plot('ankle_flex_ext', "Tobillos - FlexoExtensión")

        # Tobillos - Abducción
        plt.subplot(5, 3, 11)
        self.standard_plot('ankle_add_abd', "Tobillos - Abducción")

        # Tobillos - Rotación
        plt.subplot(5, 3, 12)
        self.standard_plot('ankle_rotation', "Tobillos - Rotación")

        # Pies - Dirección
        plt.subplot(5, 3, 15)
        label = self.standard_plot('foot_rotation', "Pies - Dirección")

        plt.subplots_adjust(left=.075, bottom=.085, right=.925,
                            top=.915, wspace=0.5, hspace=0.5)
        self.save_fig(image_name, self.data_path)
        plt.figlegend((label), ('Izquierda  ' + '(' + str(strides_left) + ')',
                                'Derecha  ' + '(' + str(strides_right) + ')'), title="# de zancadas",
                      loc='lower center', bbox_to_anchor=(0.5, 0.075),
                      frameon=False, ncol=2, fancybox=True, shadow=False)

    def add_general_info(self):
        return {
            'subject_name': self.general_info['name'],
            'report_date': datetime.datetime.now().strftime('%d/%m/%y'),
            'age': self.general_info['age'],
            'gender': self.general_info['gender'],
            'h': "{:.2f}".format(self.general_info["height"] / 100),
            'w': self.general_info["weight"],
            'leg': self.general_info["foot_length"],
            'drt': self.spatiotemporal_ovw["Gait_Time"][1],
            's': round(self.spatiotemporal_ovw["Speed"][1], 1),
            'c': round(self.spatiotemporal_ovw["Cadence"][1], 1),
            'l': int(self.spatiotemporal_ovw["Total_Strides_Left"][1]),
            'r': int(self.spatiotemporal_ovw["Total_Strides_Right"][1]),
            'l2': int(self.spatiotemporal_ovw["Processed_Strides_Left"][1]),
            'r2': int(self.spatiotemporal_ovw["Processed_Strides_Right"][1]),
            'turns': int(self.spatiotemporal_ovw["Turns"][1]),
        }

    def add_gait_cycle(self):
        return {
            'tda1m': "{:.2f}".format(self.spatiotemporal_right_cicle["double_support1_duration"].mean()),
            'tda2m': "{:.2f}".format(self.spatiotemporal_right_cicle["double_support2_duration"].mean()),
            'tda3m': "{:.2f}".format(self.spatiotemporal_left_cicle["double_support1_duration"].mean()),
            'tapsdm': "{:.2f}".format(self.spatiotemporal_right_cicle["stance_duration"].mean()
                                      - (self.spatiotemporal_right_cicle["double_support1_duration"].mean()
                                         + self.spatiotemporal_right_cicle["double_support2_duration"].mean())),
            'tapsim': "{:.2f}".format(self.spatiotemporal_left_cicle["stance_duration"].mean()
                                      - (self.spatiotemporal_left_cicle["double_support2_duration"].mean()
                                         + self.spatiotemporal_left_cicle["double_support1_duration"].mean())),
            'fbim': "{:.2f}".format(self.spatiotemporal_left_cicle["swing_percent"].mean()),
            'fbim2': "{:.2f}".format(self.spatiotemporal_left_cicle["swing_percent"].median()),
            'fbistd': "{:.2f}".format(self.spatiotemporal_left_cicle["swing_percent"].std()),
            'fbimin': "{:.2f}".format(self.spatiotemporal_left_cicle["swing_percent"].min()),
            'fbimax': "{:.2f}".format(self.spatiotemporal_left_cicle["swing_percent"].max()),
            'fbdm': "{:.2f}".format(self.spatiotemporal_right_cicle["swing_percent"].mean()),
            'fbdm2': "{:.2f}".format(self.spatiotemporal_right_cicle["swing_percent"].median()),
            'fbdstd': "{:.2f}".format(self.spatiotemporal_right_cicle["swing_percent"].std()),
            'fbdmin': "{:.2f}".format(self.spatiotemporal_right_cicle["swing_percent"].min()),
            'fbdmax': "{:.2f}".format(self.spatiotemporal_right_cicle["swing_percent"].max()),
            'rcim': "{:.2f}".format(self.spatiotemporal_left_cicle["loading_end"].mean()),
            'rcim2': "{:.2f}".format(self.spatiotemporal_left_cicle["loading_end"].median()),
            'rcistd': "{:.2f}".format(self.spatiotemporal_left_cicle["loading_end"].std()),
            'rcimin': "{:.2f}".format(self.spatiotemporal_left_cicle["loading_end"].min()),
            'rcimax': "{:.2f}".format(self.spatiotemporal_left_cicle["loading_end"].max()),
            'rcdm': "{:.2f}".format(self.spatiotemporal_right_cicle["loading_end"].mean()),
            'rcdm2': "{:.2f}".format(self.spatiotemporal_right_cicle["loading_end"].median()),
            'rcdstd': "{:.2f}".format(self.spatiotemporal_right_cicle["loading_end"].std()),
            'rcdmin': "{:.2f}".format(self.spatiotemporal_right_cicle["loading_end"].min()),
            'rcdmax': "{:.2f}".format(self.spatiotemporal_right_cicle["loading_end"].max()),
            'ipim': "{:.2f}".format(self.spatiotemporal_left_cicle["preswing_init"].mean()),
            'ipim2': "{:.2f}".format(self.spatiotemporal_left_cicle["preswing_init"].median()),
            'ipistd': "{:.2f}".format(self.spatiotemporal_left_cicle["preswing_init"].std()),
            'ipimin': "{:.2f}".format(self.spatiotemporal_left_cicle["preswing_init"].min()),
            'ipimax': "{:.2f}".format(self.spatiotemporal_left_cicle["preswing_init"].max()),
            'ipdm': "{:.2f}".format(self.spatiotemporal_right_cicle["preswing_init"].mean()),
            'ipdm2': "{:.2f}".format(self.spatiotemporal_right_cicle["preswing_init"].median()),
            'ipdstd': "{:.2f}".format(self.spatiotemporal_right_cicle["preswing_init"].std()),
            'ipdmin': "{:.2f}".format(self.spatiotemporal_right_cicle["preswing_init"].min()),
            'ipdmax': "{:.2f}".format(self.spatiotemporal_right_cicle["preswing_init"].max()),
            'faim': "{:.2f}".format(self.spatiotemporal_left_cicle["stance_percent"].mean()),
            'faim2': "{:.2f}".format(self.spatiotemporal_left_cicle["stance_percent"].median()),
            'faistd': "{:.2f}".format(self.spatiotemporal_left_cicle["stance_percent"].std()),
            'faimin': "{:.2f}".format(self.spatiotemporal_left_cicle["stance_percent"].min()),
            'faimax': "{:.2f}".format(self.spatiotemporal_left_cicle["stance_percent"].max()),
            'fadm': "{:.2f}".format(self.spatiotemporal_right_cicle["stance_percent"].mean()),
            'fadm2': "{:.2f}".format(self.spatiotemporal_right_cicle["stance_percent"].median()),
            'fadstd': "{:.2f}".format(self.spatiotemporal_right_cicle["stance_percent"].std()),
            'fadmin': "{:.2f}".format(self.spatiotemporal_right_cicle["stance_percent"].min()),
            'fadmax': "{:.2f}".format(self.spatiotemporal_right_cicle["stance_percent"].max()),
            'dbim': "{:.2f}".format(self.spatiotemporal_left_cicle["double_support_percent"].mean()),
            'dbim2': "{:.2f}".format(self.spatiotemporal_left_cicle["double_support_percent"].median()),
            'dbistd': "{:.2f}".format(self.spatiotemporal_left_cicle["double_support_percent"].std()),
            'dbimin': "{:.2f}".format(self.spatiotemporal_left_cicle["double_support_percent"].min()),
            'dbimax': "{:.2f}".format(self.spatiotemporal_left_cicle["double_support_percent"].max()),
            'dbdm': "{:.2f}".format(self.spatiotemporal_right_cicle["double_support_percent"].mean()),
            'dbdm2': "{:.2f}".format(self.spatiotemporal_right_cicle["double_support_percent"].median()),
            'dbdstd': "{:.2f}".format(self.spatiotemporal_right_cicle["double_support_percent"].std()),
            'dbdmin': "{:.2f}".format(self.spatiotemporal_right_cicle["double_support_percent"].min()),
            'dbdmax': "{:.2f}".format(self.spatiotemporal_right_cicle["double_support_percent"].max()),

            # Parámetros espaciotemporales - Análisis Detallado - Temporales
            'dzim': "{:.2f}".format(self.spatiotemporal_left_cicle["stride_duration"].mean()),
            'dzim2': "{:.2f}".format(self.spatiotemporal_left_cicle["stride_duration"].median()),
            'dzistd': "{:.2f}".format(self.spatiotemporal_left_cicle["stride_duration"].std()),
            'dzimin': "{:.2f}".format(self.spatiotemporal_left_cicle["stride_duration"].min()),
            'dzimax': "{:.2f}".format(self.spatiotemporal_left_cicle["stride_duration"].max()),
            'dzdm': "{:.2f}".format(self.spatiotemporal_right_cicle["stride_duration"].mean()),
            'dzdm2': "{:.2f}".format(self.spatiotemporal_right_cicle["stride_duration"].median()),
            'dzdstd': "{:.2f}".format(self.spatiotemporal_right_cicle["stride_duration"].std()),
            'dzdmin': "{:.2f}".format(self.spatiotemporal_right_cicle["stride_duration"].min()),
            'dzdmax': "{:.2f}".format(self.spatiotemporal_right_cicle["stride_duration"].max()),
            'dpim': "{:.2f}".format(self.spatiotemporal_left_cicle["step_duration"].mean()),
            'dpim2': "{:.2f}".format(self.spatiotemporal_left_cicle["step_duration"].median()),
            'dpistd': "{:.2f}".format(self.spatiotemporal_left_cicle["step_duration"].std()),
            'dpimin': "{:.2f}".format(self.spatiotemporal_left_cicle["step_duration"].min()),
            'dpimax': "{:.2f}".format(self.spatiotemporal_left_cicle["step_duration"].max()),
            'dpdm': "{:.2f}".format(self.spatiotemporal_right_cicle["step_duration"].mean()),
            'dpdm2': "{:.2f}".format(self.spatiotemporal_right_cicle["step_duration"].median()),
            'dpdstd': "{:.2f}".format(self.spatiotemporal_right_cicle["step_duration"].std()),
            'dpdmin': "{:.2f}".format(self.spatiotemporal_right_cicle["step_duration"].min()),
            'dpdmax': "{:.2f}".format(self.spatiotemporal_right_cicle["step_duration"].max()),
            'dfbim': "{:.2f}".format(self.spatiotemporal_left_cicle["swing_duration"].mean()),
            'dfbim2': "{:.2f}".format(self.spatiotemporal_left_cicle["swing_duration"].median()),
            'dfbistd': "{:.2f}".format(self.spatiotemporal_left_cicle["swing_duration"].std()),
            'dfbimin': "{:.2f}".format(self.spatiotemporal_left_cicle["swing_duration"].min()),
            'dfbimax': "{:.2f}".format(self.spatiotemporal_left_cicle["swing_duration"].max()),
            'dfbdm': "{:.2f}".format(self.spatiotemporal_right_cicle["swing_duration"].mean()),
            'dfbdm2': "{:.2f}".format(self.spatiotemporal_right_cicle["swing_duration"].median()),
            'dfbdstd': "{:.2f}".format(self.spatiotemporal_right_cicle["swing_duration"].std()),
            'dfbdmin': "{:.2f}".format(self.spatiotemporal_right_cicle["swing_duration"].min()),
            'dfbdmax': "{:.2f}".format(self.spatiotemporal_right_cicle["swing_duration"].max()),
            'dfaim': "{:.2f}".format(self.spatiotemporal_left_cicle["stance_duration"].mean()),
            'dfaim2': "{:.2f}".format(self.spatiotemporal_left_cicle["stance_duration"].median()),
            'dfaistd': "{:.2f}".format(self.spatiotemporal_left_cicle["stance_duration"].std()),
            'dfaimin': "{:.2f}".format(self.spatiotemporal_left_cicle["stance_duration"].min()),
            'dfaimax': "{:.2f}".format(self.spatiotemporal_left_cicle["stance_duration"].max()),
            'dfadm': "{:.2f}".format(self.spatiotemporal_right_cicle["stance_duration"].mean()),
            'dfadm2': "{:.2f}".format(self.spatiotemporal_right_cicle["stance_duration"].median()),
            'dfadstd': "{:.2f}".format(self.spatiotemporal_right_cicle["stance_duration"].std()),
            'dfadmin': "{:.2f}".format(self.spatiotemporal_right_cicle["stance_duration"].min()),
            'dfadmax': "{:.2f}".format(self.spatiotemporal_right_cicle["stance_duration"].max()),
            'tdbim': "{:.2f}".format(self.spatiotemporal_left_cicle["double_support_duration"].mean()),
            'tdbim2': "{:.2f}".format(self.spatiotemporal_left_cicle["double_support_duration"].median()),
            'tdbistd': "{:.2f}".format(self.spatiotemporal_left_cicle["double_support_duration"].std()),
            'tdbimin': "{:.2f}".format(self.spatiotemporal_left_cicle["double_support_duration"].min()),
            'tdbimax': "{:.2f}".format(self.spatiotemporal_left_cicle["double_support_duration"].max()),
            'tdbdm': "{:.2f}".format(self.spatiotemporal_right_cicle["double_support_duration"].mean()),
            'tdbdm2': "{:.2f}".format(self.spatiotemporal_right_cicle["double_support_duration"].median()),
            'tdbdstd': "{:.2f}".format(self.spatiotemporal_right_cicle["double_support_duration"].std()),
            'tdbdmin': "{:.2f}".format(self.spatiotemporal_right_cicle["double_support_duration"].min()),
            'tdbdmax': "{:.2f}".format(self.spatiotemporal_right_cicle["double_support_duration"].max()),

            # Parámetros espaciotemporales - Gráfica 1
            'lzd': "{:.2f}".format(self.spatiotemporal_ovw["Right_Stride_Length"][1] / 1000),
            'lpi': "{:.2f}".format(self.spatiotemporal_ovw["Left_Step_Length"][1] / 1000),
            'lpd': "{:.2f}".format(self.spatiotemporal_ovw["Right_Step_Length"][1] / 1000),
            'lzi': "{:.2f}".format(self.spatiotemporal_ovw["Left_Stride_Length"][1] / 1000),
            'dzd': "{:.2f}".format(self.spatiotemporal_ovw["Right_Stride_Duration"][1]),
            'dpi': "{:.2f}".format(self.spatiotemporal_ovw["Left_Step_Duration"][1]),
            'dpd': "{:.2f}".format(self.spatiotemporal_ovw["Right_Step_Duration"][1]),
            'dzi': "{:.2f}".format(self.spatiotemporal_ovw["Left_Stride_Duration"][1]),
            'ap': "{:.2f}".format(self.spatiotemporal_ovw["Base_Of_Support"][1] / 1000),
            'angi': "{:.2f}".format(self.spatiotemporal_ovw["Left_Step_Angle"][1]),
            'angd': "{:.2f}".format(self.spatiotemporal_ovw["Right_Step_Angle"][1]),
        }

    def add_spatial_info(self):
        return {
            # Parámetros espaciotemporales - Análisis Detallado - Espaciales
            'lzim': "{:.2f}".format(self.spatiotemporal_left_spatial["stride_length"].mean()),
            'lzim2': "{:.2f}".format(self.spatiotemporal_left_spatial["stride_length"].median()),
            'lzistd': "{:.2f}".format(self.spatiotemporal_left_spatial["stride_length"].std()),
            'lzimin': "{:.2f}".format(self.spatiotemporal_left_spatial["stride_length"].min()),
            'lzimax': "{:.2f}".format(self.spatiotemporal_left_spatial["stride_length"].max()),
            'lzdm': "{:.2f}".format(self.spatiotemporal_right_spatial["stride_length"].mean()),
            'lzdm2': "{:.2f}".format(self.spatiotemporal_right_spatial["stride_length"].median()),
            'lzdstd': "{:.2f}".format(self.spatiotemporal_right_spatial["stride_length"].std()),
            'lzdmin': "{:.2f}".format(self.spatiotemporal_right_spatial["stride_length"].min()),
            'lzdmax': "{:.2f}".format(self.spatiotemporal_right_spatial["stride_length"].max()),
            'lpim': "{:.2f}".format(self.spatiotemporal_left_spatial["step_length"].mean()),
            'lpim2': "{:.2f}".format(self.spatiotemporal_left_spatial["step_length"].median()),
            'lpistd': "{:.2f}".format(self.spatiotemporal_left_spatial["step_length"].std()),
            'lpimin': "{:.2f}".format(self.spatiotemporal_left_spatial["step_length"].min()),
            'lpimax': "{:.2f}".format(self.spatiotemporal_left_spatial["step_length"].max()),
            'lpdm': "{:.2f}".format(self.spatiotemporal_right_spatial["step_length"].mean()),
            'lpdm2': "{:.2f}".format(self.spatiotemporal_right_spatial["step_length"].median()),
            'lpdstd': "{:.2f}".format(self.spatiotemporal_right_spatial["step_length"].std()),
            'lpdmin': "{:.2f}".format(self.spatiotemporal_right_spatial["step_length"].min()),
            'lpdmax': "{:.2f}".format(self.spatiotemporal_right_spatial["step_length"].max()),
            'apim': "{:.2f}".format(self.spatiotemporal_left_spatial["base_of_support"].mean()),
            'apim2': "{:.2f}".format(self.spatiotemporal_left_spatial["base_of_support"].median()),
            'apistd': "{:.2f}".format(self.spatiotemporal_left_spatial["base_of_support"].std()),
            'apimin': "{:.2f}".format(self.spatiotemporal_left_spatial["base_of_support"].min()),
            'apimax': "{:.2f}".format(self.spatiotemporal_left_spatial["base_of_support"].max()),
            'apdm': "{:.2f}".format(self.spatiotemporal_right_spatial["base_of_support"].mean()),
            'apdm2': "{:.2f}".format(self.spatiotemporal_right_spatial["base_of_support"].median()),
            'apdstd': "{:.2f}".format(self.spatiotemporal_right_spatial["base_of_support"].std()),
            'apdmin': "{:.2f}".format(self.spatiotemporal_right_spatial["base_of_support"].min()),
            'apdmax': "{:.2f}".format(self.spatiotemporal_right_spatial["base_of_support"].max()),
            'anglim': "{:.2f}".format(self.spatiotemporal_left_spatial["step_angle"].mean()),
            'anglim2': "{:.2f}".format(self.spatiotemporal_left_spatial["step_angle"].median()),
            'anglistd': "{:.2f}".format(self.spatiotemporal_left_spatial["step_angle"].std()),
            'anglimin': "{:.2f}".format(self.spatiotemporal_left_spatial["step_angle"].min()),
            'anglimax': "{:.2f}".format(self.spatiotemporal_left_spatial["step_angle"].max()),
            'angldm': "{:.2f}".format(self.spatiotemporal_right_spatial["step_angle"].mean()),
            'angldm2': "{:.2f}".format(self.spatiotemporal_right_spatial["step_angle"].median()),
            'angldstd': "{:.2f}".format(self.spatiotemporal_right_spatial["step_angle"].std()),
            'angldmin': "{:.2f}".format(self.spatiotemporal_right_spatial["step_angle"].min()),
            'angldmax': "{:.2f}".format(self.spatiotemporal_right_spatial["step_angle"].max()),
            'matim': "{:.2f}".format(self.spatiotemporal_left_spatial["max_heel_height"].mean()),
            'matim2': "{:.2f}".format(self.spatiotemporal_left_spatial["max_heel_height"].median()),
            'matistd': "{:.2f}".format(self.spatiotemporal_left_spatial["max_heel_height"].std()),
            'matimin': "{:.2f}".format(self.spatiotemporal_left_spatial["max_heel_height"].min()),
            'matimax': "{:.2f}".format(self.spatiotemporal_left_spatial["max_heel_height"].max()),
            'matdm': "{:.2f}".format(self.spatiotemporal_right_spatial["max_heel_height"].mean()),
            'matdm2': "{:.2f}".format(self.spatiotemporal_right_spatial["max_heel_height"].median()),
            'matdstd': "{:.2f}".format(self.spatiotemporal_right_spatial["max_heel_height"].std()),
            'matdmin': "{:.2f}".format(self.spatiotemporal_right_spatial["max_heel_height"].min()),
            'matdmax': "{:.2f}".format(self.spatiotemporal_right_spatial["max_heel_height"].max()),
            'mapim': "{:.2f}".format(self.spatiotemporal_left_spatial["max_toe_height"].mean()),
            'mapim2': "{:.2f}".format(self.spatiotemporal_left_spatial["max_toe_height"].median()),
            'mapistd': "{:.2f}".format(self.spatiotemporal_left_spatial["max_toe_height"].std()),
            'mapimin': "{:.2f}".format(self.spatiotemporal_left_spatial["max_toe_height"].min()),
            'mapimax': "{:.2f}".format(self.spatiotemporal_left_spatial["max_toe_height"].max()),
            'mapdm': "{:.2f}".format(self.spatiotemporal_right_spatial["max_toe_height"].mean()),
            'mapdm2': "{:.2f}".format(self.spatiotemporal_right_spatial["max_toe_height"].median()),
            'mapdstd': "{:.2f}".format(self.spatiotemporal_right_spatial["max_toe_height"].std()),
            'mapdmin': "{:.2f}".format(self.spatiotemporal_right_spatial["max_toe_height"].min()),
            'mapdmax': "{:.2f}".format(self.spatiotemporal_right_spatial["max_toe_height"].max()),
            'angcim': "{:.2f}".format(self.spatiotemporal_left_spatial["strike_angle"].mean()),
            'angcim2': "{:.2f}".format(self.spatiotemporal_left_spatial["strike_angle"].median()),
            'angcistd': "{:.2f}".format(self.spatiotemporal_left_spatial["strike_angle"].std()),
            'angcimin': "{:.2f}".format(self.spatiotemporal_left_spatial["strike_angle"].min()),
            'angcimax': "{:.2f}".format(self.spatiotemporal_left_spatial["strike_angle"].max()),
            'angcdm': "{:.2f}".format(self.spatiotemporal_right_spatial["strike_angle"].mean()),
            'angcdm2': "{:.2f}".format(self.spatiotemporal_right_spatial["strike_angle"].median()),
            'angcdstd': "{:.2f}".format(self.spatiotemporal_right_spatial["strike_angle"].std()),
            'angcdmin': "{:.2f}".format(self.spatiotemporal_right_spatial["strike_angle"].min()),
            'angcdmax': "{:.2f}".format(self.spatiotemporal_right_spatial["strike_angle"].max()),
            'angdim': "{:.2f}".format(self.spatiotemporal_left_spatial["toe_off_angle"].mean()),
            'angdim2': "{:.2f}".format(self.spatiotemporal_left_spatial["toe_off_angle"].median()),
            'angdistd': "{:.2f}".format(self.spatiotemporal_left_spatial["toe_off_angle"].std()),
            'angdimin': "{:.2f}".format(self.spatiotemporal_left_spatial["toe_off_angle"].min()),
            'angdimax': "{:.2f}".format(self.spatiotemporal_left_spatial["toe_off_angle"].max()),
            'angddm': "{:.2f}".format(self.spatiotemporal_right_spatial["toe_off_angle"].mean()),
            'angddm2': "{:.2f}".format(self.spatiotemporal_right_spatial["toe_off_angle"].median()),
            'angddstd': "{:.2f}".format(self.spatiotemporal_right_spatial["toe_off_angle"].std()),
            'angddmin': "{:.2f}".format(self.spatiotemporal_right_spatial["toe_off_angle"].min()),
            'angddmax': "{:.2f}".format(self.spatiotemporal_right_spatial["toe_off_angle"].max()),


            # Parámetros espaciotemporales - Análisis Detallado - Desplazamiento CDG
            'cdgvim': "{:.2f}".format(self.spatiotemporal_left_spatial["com_vertical_displacement"].mean()),
            'cdgvim2': "{:.2f}".format(self.spatiotemporal_left_spatial["com_vertical_displacement"].median()),
            'cdgvistd': "{:.2f}".format(self.spatiotemporal_left_spatial["com_vertical_displacement"].std()),
            'cdgvimin': "{:.2f}".format(self.spatiotemporal_left_spatial["com_vertical_displacement"].min()),
            'cdgvimax': "{:.2f}".format(self.spatiotemporal_left_spatial["com_vertical_displacement"].max()),
            'cdgvdm': "{:.2f}".format(self.spatiotemporal_right_spatial["com_vertical_displacement"].mean()),
            'cdgvdm2': "{:.2f}".format(self.spatiotemporal_right_spatial["com_vertical_displacement"].median()),
            'cdgvdstd': "{:.2f}".format(self.spatiotemporal_right_spatial["com_vertical_displacement"].std()),
            'cdgvdmin': "{:.2f}".format(self.spatiotemporal_right_spatial["com_vertical_displacement"].min()),
            'cdgvdmax': "{:.2f}".format(self.spatiotemporal_right_spatial["com_vertical_displacement"].max()),
            'cdghim': "{:.2f}".format(self.spatiotemporal_left_spatial["com_horizontal_displacement"].mean()),
            'cdghim2': "{:.2f}".format(self.spatiotemporal_left_spatial["com_horizontal_displacement"].median()),
            'cdghistd': "{:.2f}".format(self.spatiotemporal_left_spatial["com_horizontal_displacement"].std()),
            'cdghimin': "{:.2f}".format(self.spatiotemporal_left_spatial["com_horizontal_displacement"].min()),
            'cdghimax': "{:.2f}".format(self.spatiotemporal_left_spatial["com_horizontal_displacement"].max()),
            'cdghdm': "{:.2f}".format(self.spatiotemporal_right_spatial["com_horizontal_displacement"].mean()),
            'cdghdm2': "{:.2f}".format(self.spatiotemporal_right_spatial["com_horizontal_displacement"].median()),
            'cdghdstd': "{:.2f}".format(self.spatiotemporal_right_spatial["com_horizontal_displacement"].std()),
            'cdghdmin': "{:.2f}".format(self.spatiotemporal_right_spatial["com_horizontal_displacement"].min()),
            'cdghdmax': "{:.2f}".format(self.spatiotemporal_right_spatial["com_horizontal_displacement"].max()),

            'cdgvm': "{:.2f}".format((self.spatiotemporal_left_spatial["com_vertical_displacement"].mean()
                                      + self.spatiotemporal_right_spatial["com_vertical_displacement"].mean()) / 2),
            'cdghm': "{:.2f}".format((self.spatiotemporal_left_spatial["com_horizontal_displacement"].mean()
                                      + self.spatiotemporal_right_spatial["com_horizontal_displacement"].mean()) / 2),
        }

    def add_kinematics_info(self):
        return {
            # Análisis cinemático - Pelvis (Inclinación)
            'plviim': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_pelvis_tilt', 'Mean'].values[0]),
            'plviimin': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_pelvis_tilt', 'Min'].values[0]),
            'plviimax': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_pelvis_tilt', 'Max'].values[0]),
            'plviir': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_pelvis_tilt', 'Range'].values[0]),
            'plviii': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_pelvis_tilt', 'Stride_Start'].values[0]),
            'plviif':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_pelvis_tilt', 'Stride_End'].values[0]),
            'plviit':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_pelvis_tilt', 'Swing_Angle'].values[0]),
            'plvidm': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_pelvis_tilt', 'Mean'].values[0]),
            'plvidmin': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_pelvis_tilt', 'Min'].values[0]),
            'plvidmax': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_pelvis_tilt', 'Max'].values[0]),
            'plvidr': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_pelvis_tilt', 'Range'].values[0]),
            'plvidi': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_pelvis_tilt', 'Stride_Start'].values[0]),
            'plvidf':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_pelvis_tilt', 'Stride_End'].values[0]),
            'plvidt':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_pelvis_tilt', 'Swing_Angle'].values[0]),

            # Análisis cinemático - Pelvis (Oblicuidad)
            'plvoim': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_pelvis_obliquity', 'Mean'].values[0]),
            'plvoimin': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_pelvis_obliquity', 'Min'].values[0]),
            'plvoimax': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_pelvis_obliquity', 'Max'].values[0]),
            'plvoir': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_pelvis_obliquity', 'Range'].values[0]),
            'plvoii': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_pelvis_obliquity', 'Stride_Start'].values[0]),
            'plvoif':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_pelvis_obliquity', 'Stride_End'].values[0]),
            'plvoit':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_pelvis_obliquity', 'Swing_Angle'].values[0]),
            'plvodm': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_pelvis_obliquity', 'Mean'].values[0]),
            'plvodmin': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_pelvis_obliquity', 'Min'].values[0]),
            'plvodmax': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_pelvis_obliquity', 'Max'].values[0]),
            'plvodr': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_pelvis_obliquity', 'Range'].values[0]),
            'plvodi': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_pelvis_obliquity', 'Stride_Start'].values[0]),
            'plvodf':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_pelvis_obliquity', 'Stride_End'].values[0]),
            'plvodt':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_pelvis_obliquity', 'Swing_Angle'].values[0]),

            # Análisis cinemático - Pelvis (Rotación)
            'plvrim': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_pelvis_rotation', 'Mean'].values[0]),
            'plvrimin': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_pelvis_rotation', 'Min'].values[0]),
            'plvrimax': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_pelvis_rotation', 'Max'].values[0]),
            'plvrir': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_pelvis_rotation', 'Range'].values[0]),
            'plvrii': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_pelvis_rotation', 'Stride_Start'].values[0]),
            'plvrif':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_pelvis_rotation', 'Stride_End'].values[0]),
            'plvrit':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_pelvis_rotation', 'Swing_Angle'].values[0]),
            'plvrdm': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_pelvis_rotation', 'Mean'].values[0]),
            'plvrdmin': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_pelvis_rotation', 'Min'].values[0]),
            'plvrdmax': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_pelvis_rotation', 'Max'].values[0]),
            'plvrdr': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_pelvis_rotation', 'Range'].values[0]),
            'plvrdi': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_pelvis_rotation', 'Stride_Start'].values[0]),
            'plvrdf':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_pelvis_rotation', 'Stride_End'].values[0]),
            'plvrdt':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_pelvis_rotation', 'Swing_Angle'].values[0]),

            # Análisis cinemático - Cadera (Flexo Extensión)
            'hipfeim': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_hip_flex_ext', 'Mean'].values[0]),
            'hipfeimin': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_hip_flex_ext', 'Min'].values[0]),
            'hipfeimax': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_hip_flex_ext', 'Max'].values[0]),
            'hipfeir': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_hip_flex_ext', 'Range'].values[0]),
            'hipfeii': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_hip_flex_ext', 'Stride_Start'].values[0]),
            'hipfeif':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_hip_flex_ext', 'Stride_End'].values[0]),
            'hipfeit':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_hip_flex_ext', 'Swing_Angle'].values[0]),
            'hipfedm': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_hip_flex_ext', 'Mean'].values[0]),
            'hipfedmin': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_hip_flex_ext', 'Min'].values[0]),
            'hipfedmax': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_hip_flex_ext', 'Max'].values[0]),
            'hipfedr': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_hip_flex_ext', 'Range'].values[0]),
            'hipfedi': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_hip_flex_ext', 'Stride_Start'].values[0]),
            'hipfedf':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_hip_flex_ext', 'Stride_End'].values[0]),
            'hipfedt':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_hip_flex_ext', 'Swing_Angle'].values[0]),


            # Análisis cinemático - Cadera (Abducción Aducción)
            'hipaaim': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_hip_add_abd', 'Mean'].values[0]),
            'hipaaimin': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_hip_add_abd', 'Min'].values[0]),
            'hipaaimax': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_hip_add_abd', 'Max'].values[0]),
            'hipaair': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_hip_add_abd', 'Range'].values[0]),
            'hipaaii': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_hip_add_abd', 'Stride_Start'].values[0]),
            'hipaaif':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_hip_add_abd', 'Stride_End'].values[0]),
            'hipaait':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_hip_add_abd', 'Swing_Angle'].values[0]),
            'hipaadm': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_hip_add_abd', 'Mean'].values[0]),
            'hipaadmin': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_hip_add_abd', 'Min'].values[0]),
            'hipaadmax': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_hip_add_abd', 'Max'].values[0]),
            'hipaadr': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_hip_add_abd', 'Range'].values[0]),
            'hipaadi': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_hip_add_abd', 'Stride_Start'].values[0]),
            'hipaadf':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_hip_add_abd', 'Stride_End'].values[0]),
            'hipaadt':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_hip_add_abd', 'Swing_Angle'].values[0]),


            # Análisis cinemático - Cadera (Rotación)
            'hiprim': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_hip_rotation', 'Mean'].values[0]),
            'hiprimin': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_hip_rotation', 'Min'].values[0]),
            'hiprimax': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_hip_rotation', 'Max'].values[0]),
            'hiprir': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_hip_rotation', 'Range'].values[0]),
            'hiprii': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_hip_rotation', 'Stride_Start'].values[0]),
            'hiprif':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_hip_rotation', 'Stride_End'].values[0]),
            'hiprit':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_hip_rotation', 'Swing_Angle'].values[0]),
            'hiprdm': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_hip_rotation', 'Mean'].values[0]),
            'hiprdmin': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_hip_rotation', 'Min'].values[0]),
            'hiprdmax': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_hip_rotation', 'Max'].values[0]),
            'hiprdr': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_hip_rotation', 'Range'].values[0]),
            'hiprdi': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_hip_rotation', 'Stride_Start'].values[0]),
            'hiprdf':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_hip_rotation', 'Stride_End'].values[0]),
            'hiprdt':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_hip_rotation', 'Swing_Angle'].values[0]),

            # Análisis cinemático - Rodilla (Flexo Extensión)
            'knefeim': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_knee_flex_ext', 'Mean'].values[0]),
            'knefeimin': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_knee_flex_ext', 'Min'].values[0]),
            'knefeimax': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_knee_flex_ext', 'Max'].values[0]),
            'knefeir': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_knee_flex_ext', 'Range'].values[0]),
            'knefeii': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_knee_flex_ext', 'Stride_Start'].values[0]),
            'knefeif':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_knee_flex_ext', 'Stride_End'].values[0]),
            'knefeit':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_knee_flex_ext', 'Swing_Angle'].values[0]),
            'knefedm': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_knee_flex_ext', 'Mean'].values[0]),
            'knefedmin': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_knee_flex_ext', 'Min'].values[0]),
            'knefedmax': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_knee_flex_ext', 'Max'].values[0]),
            'knefedr': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_knee_flex_ext', 'Range'].values[0]),
            'knefedi': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_knee_flex_ext', 'Stride_Start'].values[0]),
            'knefedf':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_knee_flex_ext', 'Stride_End'].values[0]),
            'knefedt':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_knee_flex_ext', 'Swing_Angle'].values[0]),


            # Análisis cinemático - Rodilla (Abducción Aducción)
            'kneaaim': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_knee_val_var', 'Mean'].values[0]),
            'kneaaimin': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_knee_val_var', 'Min'].values[0]),
            'kneaaimax': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_knee_val_var', 'Max'].values[0]),
            'kneaair': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_knee_val_var', 'Range'].values[0]),
            'kneaaii': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_knee_val_var', 'Stride_Start'].values[0]),
            'kneaaif':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_knee_val_var', 'Stride_End'].values[0]),
            'kneaait':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_knee_val_var', 'Swing_Angle'].values[0]),
            'kneaadm': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_knee_val_var', 'Mean'].values[0]),
            'kneaadmin': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_knee_val_var', 'Min'].values[0]),
            'kneaadmax': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_knee_val_var', 'Max'].values[0]),
            'kneaadr': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_knee_val_var', 'Range'].values[0]),
            'kneaadi': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_knee_val_var', 'Stride_Start'].values[0]),
            'kneaadf':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_knee_val_var', 'Stride_End'].values[0]),
            'kneaadt':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_knee_val_var', 'Swing_Angle'].values[0]),


            # Análisis cinemático - Rodilla (Rotación)
            'knerim': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_knee_rotation', 'Mean'].values[0]),
            'knerimin': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_knee_rotation', 'Min'].values[0]),
            'knerimax': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_knee_rotation', 'Max'].values[0]),
            'knerir': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_knee_rotation', 'Range'].values[0]),
            'knerii': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_knee_rotation', 'Stride_Start'].values[0]),
            'knerif':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_knee_rotation', 'Stride_End'].values[0]),
            'knerit':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_knee_rotation', 'Swing_Angle'].values[0]),
            'knerdm': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_knee_rotation', 'Mean'].values[0]),
            'knerdmin': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_knee_rotation', 'Min'].values[0]),
            'knerdmax': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_knee_rotation', 'Max'].values[0]),
            'knerdr': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_knee_rotation', 'Range'].values[0]),
            'knerdi': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_knee_rotation', 'Stride_Start'].values[0]),
            'knerdf':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_knee_rotation', 'Stride_End'].values[0]),
            'knerdt':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_knee_rotation', 'Swing_Angle'].values[0]),

            # Análisis cinemático - Tobillo (Flexo Extensión)
            'ankfeim': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_ankle_flex_ext', 'Mean'].values[0]),
            'ankfeimin': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_ankle_flex_ext', 'Min'].values[0]),
            'ankfeimax': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_ankle_flex_ext', 'Max'].values[0]),
            'ankfeir': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_ankle_flex_ext', 'Range'].values[0]),
            'ankfeii': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_ankle_flex_ext', 'Stride_Start'].values[0]),
            'ankfeif':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_ankle_flex_ext', 'Stride_End'].values[0]),
            'ankfeit':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_ankle_flex_ext', 'Swing_Angle'].values[0]),
            'ankfedm': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_ankle_flex_ext', 'Mean'].values[0]),
            'ankfedmin': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_ankle_flex_ext', 'Min'].values[0]),
            'ankfedmax': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_ankle_flex_ext', 'Max'].values[0]),
            'ankfedr': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_ankle_flex_ext', 'Range'].values[0]),
            'ankfedi': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_ankle_flex_ext', 'Stride_Start'].values[0]),
            'ankfedf':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_ankle_flex_ext', 'Stride_End'].values[0]),
            'ankfedt':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_ankle_flex_ext', 'Swing_Angle'].values[0]),

            # Análisis cinemático - Tobillo (Abducción Aducción)
            'ankaaim': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_ankle_add_abd', 'Mean'].values[0]),
            'ankaaimin': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_ankle_add_abd', 'Min'].values[0]),
            'ankaaimax': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_ankle_add_abd', 'Max'].values[0]),
            'ankaair': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_ankle_add_abd', 'Range'].values[0]),
            'ankaaii': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_ankle_add_abd', 'Stride_Start'].values[0]),
            'ankaaif':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_ankle_add_abd', 'Stride_End'].values[0]),
            'ankaait':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_ankle_add_abd', 'Swing_Angle'].values[0]),
            'ankaadm': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_ankle_add_abd', 'Mean'].values[0]),
            'ankaadmin': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_ankle_add_abd', 'Min'].values[0]),
            'ankaadmax': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_ankle_add_abd', 'Max'].values[0]),
            'ankaadr': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_ankle_add_abd', 'Range'].values[0]),
            'ankaadi': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_ankle_add_abd', 'Stride_Start'].values[0]),
            'ankaadf':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_ankle_add_abd', 'Stride_End'].values[0]),
            'ankaadt':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_ankle_add_abd', 'Swing_Angle'].values[0]),


            # Análisis cinemático - Tobillo (Rotación)
            'ankrim': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_ankle_rotation', 'Mean'].values[0]),
            'ankrimin': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_ankle_rotation', 'Min'].values[0]),
            'ankrimax': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_ankle_rotation', 'Max'].values[0]),
            'ankrir': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_ankle_rotation', 'Range'].values[0]),
            'ankrii': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_ankle_rotation', 'Stride_Start'].values[0]),
            'ankrif':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_ankle_rotation', 'Stride_End'].values[0]),
            'ankrit':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_ankle_rotation', 'Swing_Angle'].values[0]),
            'ankrdm': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_ankle_rotation', 'Mean'].values[0]),
            'ankrdmin': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_ankle_rotation', 'Min'].values[0]),
            'ankrdmax': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_ankle_rotation', 'Max'].values[0]),
            'ankrdr': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_ankle_rotation', 'Range'].values[0]),
            'ankrdi': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_ankle_rotation', 'Stride_Start'].values[0]),
            'ankrdf':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_ankle_rotation', 'Stride_End'].values[0]),
            'ankrdt':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_ankle_rotation', 'Swing_Angle'].values[0]),

            # Análisis cinemático - Pie (Rotación)
            'footfeim': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_foot_rotation', 'Mean'].values[0]),
            'footfeimin': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_foot_rotation', 'Min'].values[0]),
            'footfeimax': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_foot_rotation', 'Max'].values[0]),
            'footfeir': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_foot_rotation', 'Range'].values[0]),
            'footfeii': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_foot_rotation', 'Stride_Start'].values[0]),
            'footfeif':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_foot_rotation', 'Stride_End'].values[0]),
            'footfeit':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'left_foot_rotation', 'Swing_Angle'].values[0]),
            'footfedm': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_foot_rotation', 'Mean'].values[0]),
            'footfedmin': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_foot_rotation', 'Min'].values[0]),
            'footfedmax': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_foot_rotation', 'Max'].values[0]),
            'footfedr': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_foot_rotation', 'Range'].values[0]),
            'footfedi': "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_foot_rotation', 'Stride_Start'].values[0]),
            'footfedf':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_foot_rotation', 'Stride_End'].values[0]),
            'footfedt':  "{:.1f}".format(self.kinematics_resume.loc[self.kinematics_resume.Parameter == 'right_foot_rotation', 'Swing_Angle'].values[0]),
        }

    # https://github.com/python-openxml/python-docx/issues/113
    def convert_to_pdf(self, doc):
        word = win32com.client.DispatchEx("Word.Application")
        new_name = doc.replace(".docx", r".pdf")
        worddoc = word.Documents.Open(doc)
        worddoc.SaveAs(new_name, FileFormat = 17)
        worddoc.Close()
        word.Quit()

    def generate(self):
        self.load_data()
        context = {}
        context.update(self.add_general_info())
        context.update(self.add_gait_cycle())
        context.update(self.add_spatial_info())
        context.update(self.add_kinematics_info())

        if self.poma_preds['poma_type'] == 'st':
            self.poma_preds['poma_type'] = 'espaciotemporales'
        elif self.poma_preds['poma_type'] == 'kin':
            self.poma_preds['poma_type'] = 'cinemáticas'
        elif self.poma_preds['poma_type'] == 'stkin':
            self.poma_preds['poma_type'] = 'espaciotemporales y cinemáticas'

        context.update(self.poma_preds)

        # Generate kinematics chart
        image_name = 'kinematics_resume_image'
        self.plot_kinematics(image_name)
        context['kinematics_resume_image'] = InlineImage(self.template, image_descriptor=(
            self.data_path + '/' + image_name + '.png'), width=Mm(150))

        # Render docx
        self.template.render(context)
        self.template.save(self.output_path + 'generated_report.docx')

        # Convert to PDF
        self.convert_to_pdf(os.path.join(str(pathlib.Path().resolve()), self.output_path, 'generated_report.docx'))
