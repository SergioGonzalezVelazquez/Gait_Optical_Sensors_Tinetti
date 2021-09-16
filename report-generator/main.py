#! /usr/bin/env python3
# from turns_detector import TurnsDetector


from input_data import input_data
from pprint import pprint
import os
import sys
import pandas as pd
import shutil
import json
import joblib
import tempfile
from bcolors import bcolors
from distutils.dir_util import copy_tree
from report_generator import ReportGenerator

TEMPLATE = "template.docx"

sys.path.append('../preprocessor-cli/src')
from turns_detector import TurnsDetector
from spatiotemporal_estimator import SpatioTemporalEstimator
from final_filtering import FinalFiltering
from kinematics import KinematicsProcessor


def preprocess_data(tmp_dir):
    records = {"subject": ['rec']}

    filtering_out = os.path.join(tmp_dir, "stride_filtering.xlsx")

    # Step 1: Run turns-detector
    turns_detector = TurnsDetector(tmp_dir, records, filtering_out, enable_logger=False)
    turns_detector.run()

    # Step 2: Estimate spatio-temporal parameters
    spatiotemporal_estimator = SpatioTemporalEstimator(
        tmp_dir,  records, filtering_out, enable_logger=False)
    spatiotemporal_estimator.find_outliers()

    # Step 3: Parse kinematics files
    # Load kinematics_filtering_threshold from preprocess config
    with open("../preprocessor-cli/config.json", "r") as config:
        config_file = json.load(config)
        kinematics_filtering_threshold = config_file['kinematics_filtering_threshold']

    kinematics_processor = KinematicsProcessor(
        tmp_dir, records, filtering_out, kinematics_filtering_threshold, enable_logger=False)
    kinematics_processor.run()

    # Step 4: Generate final filtering combining previous info
    FinalFiltering(filtering_out).run()

    # Step 5: Generate processed files with filtered strides
    spatiotemporal_estimator.generate_filtered_resume()
    kinematics_processor.generate_filtered_resume()


def copy_data_to_temp_dir(tmp_dir, raw_file, events_file, cog_file, kin_dir, height, foot_length):
    '''TODO'''
    subject_path = os.path.join(tmp_dir, 'subject')
    record_path = os.path.join(subject_path, 'rec')
    os.mkdir(subject_path)
    os.mkdir(record_path)

    # Copy raw data
    raw_path = os.path.join(record_path, 'subject_rec.raw')
    shutil.copy(raw_file, raw_path)

    # Copy events file
    events_path = os.path.join(record_path, 'subject_rec.events.txt')
    shutil.copy(events_file, events_path)

    # Copy cog_file
    biomechanics_path = os.path.join(record_path, 'biomechanics')
    os.mkdir(biomechanics_path)
    shutil.copy(cog_file, biomechanics_path)

    # Copy kin dir
    biomechanics_path = os.path.join(record_path, 'strides')
    copy_tree(kin_dir, biomechanics_path)

    # Create anthropometry file
    anthropometry = [{'subject': 'subject', 'leg_length': height,
                      'foot_length': foot_length}, {'leg_length': 'cm', 'foot_length': 'mm'}, ]

    anthropometry_df = pd.DataFrame(anthropometry)
    anthropometry_path = os.path.join(tmp_dir, "anthropometry.xlsx")
    writer = pd.ExcelWriter(anthropometry_path, engine='xlsxwriter')
    anthropometry_df.to_excel(writer, index=False)
    writer.save()


def predict_poma(tmp_dir, models_type):
    poma = {'poma_type': models_type}

    kin_correlated_opp_foot = [('left_pelvis_tilt_Range', 'right_pelvis_tilt_Range'),
        ('left_pelvis_tilt_Stride_End', 'right_pelvis_tilt_Stride_End'),
        ('left_pelvis_rotation_Range', 'right_pelvis_rotation_Range'),
        ('left_pelvis_tilt_Max', 'right_pelvis_tilt_Max'),
        ('left_pelvis_tilt_Stride_Start', 'right_pelvis_tilt_Stride_Start'),
        ('left_pelvis_tilt_Min', 'right_pelvis_tilt_Min'),
        ('left_pelvis_obliquity_Range', 'right_pelvis_obliquity_Range'),
        ('left_pelvis_tilt_Mean', 'right_pelvis_tilt_Mean')]

    # Load data
    if models_type == 'stkin':
        kinematics = pd.read_csv(os.path.join(tmp_dir, 'kinematics.csv'))
        spatiotemporal = pd.read_csv(os.path.join(tmp_dir, 'spatiotemporal.csv'))
        data =  pd.merge(spatiotemporal, kinematics, on=["subject", "record"])
    elif models_type == 'kin':
        data = pd.read_csv(os.path.join(tmp_dir, 'kinematics.csv'))
    elif models_type == 'st':
        data = pd.read_csv(os.path.join(tmp_dir, 'spatiotemporal.csv'))

    if models_type == 'stkin' or models_type == 'kin':
        # Crea nuevas características a partir de los movimientos pélvicos correlacionados
        for kin_pair in kin_correlated_opp_foot:
            cols = [kin_pair[0], kin_pair[1]]
            data[kin_pair[0][kin_pair[0].find("_") + 1:]] = data[cols].mean(axis=1)
            data.drop(cols, inplace=True, axis=1)

    labels = ['lap1', 'lap2', 'lap3', 'lap4', 'dc', 'pm',]
    labels_meaning = {
        'lap1': 'LAP1: ¿El pie derecho sobrepasa al izquierdo con el paso en la fase de balanceo?',
        'lap2': 'LAP2: ¿El pie izquierdo sobrepasa al derecho con el paso en la fase de balanceo?',
        'lap3': 'LAP3: ¿El pie derecho se levanta completamente del suelo con el paso en la fase del balanceo',
        'lap4': 'LAP4: ¿El pie izquierdo se levanta completamente del suelo con el paso en la fase del balanceo',
        'dc': 'DC: ¿Para o hay discontinuidad entre pasos?',
        'pm': 'PM: ¿Talones casi se tocan mientras camina?'
        
    }

    models = {}
    score = 0
    print("\n")
    for label in labels:
        print(bcolors.BOLD + labels_meaning[label] + bcolors.ENDC)
        models[label] = joblib.load("../models/" + label + "_" + models_type + ".pkl")
        predicts_proba = models[label].predict_proba(data)[0]
        poma[label + "_0"] = round(predicts_proba[0] * 100, 2)
        poma[label + "_1"] = round(predicts_proba[1] * 100, 2)
        print("0 (" + str(poma[label + "_0"]) + "%)\t\t1 (" + str(poma[label + "_1"]) + "%)")
        if (poma[label + "_0"] > poma[label + "_1"]):
            poma[label] = "0"
            poma[label + "_perc"] =  poma[label + "_0"]
            print(bcolors.OKGREEN + "Puntuación: 0"  + bcolors.ENDC)
        else:
            score+=1
            poma[label] = "1"
            poma[label + "_perc"] =  poma[label + "_1"]
            print(bcolors.WARNING + "Puntuación: 1"  + bcolors.ENDC)
        print("\n")
    
    poma["tpg"] = str(score)
    print(bcolors.BOLD + "Puntuación total: " + poma["tpg"] + "/6"  + bcolors.ENDC)
    return poma 

def main():
    '''TODO'''
    # Get patient info and script config
    answers = input_data()

    # Create tmp dir
    tmp_dir_obj = tempfile.TemporaryDirectory()
    temp_dir = tmp_dir_obj.name
    copy_data_to_temp_dir(temp_dir, answers['raw_file'], answers['events_file'], answers['cog_file'],
                                     answers['kin_path'],
                                     answers['height'], answers['foot_length'])

    # Preprocess Clinical 3DMA data
    preprocess_data(temp_dir)

    # Predict POMA
    poma_preds = predict_poma(temp_dir, answers['predictors_type'])

    # Generate report
    general_info = {
            'name': answers['name'],
            'age': answers['age'],
            'gender': answers['gender'],
            'height': answers["height"],
            'weight': answers["weight"],
            'foot_length': answers["foot_length"]
    }
    report_out = "C:/Users/Sergio/Desktop/pruebas_temp/"
    report = ReportGenerator(temp_dir, general_info, poma_preds, TEMPLATE, report_out)
    report.generate()
    
    
    # Clean temp_dir
    tmp_dir_obj.cleanup()


if __name__ == "__main__":
    main()
