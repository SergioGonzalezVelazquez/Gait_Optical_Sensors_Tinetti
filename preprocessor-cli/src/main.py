#! /usr/bin/env python3
from utils import *
from turns_detector import TurnsDetector
from kinematics import KinematicsProcessor
from spatiotemporal_estimator import SpatioTemporalEstimator
from final_filtering import FinalFiltering
import json
import sys


def load_config():
    '''TODO'''
    # Read config file
    try:
        with open("config.json", "r") as config:
            config_file = json.load(config)
            return config_file
    except OSError as e:
        logger("File config.json not found", msgType=LoggerType.FAIL)
        sys.exit()


def main():
    '''TODO'''
    logger("Starting preprocessing")
    config = load_config()

    # Step 1: Run turns-detector
    turns_detector = TurnsDetector(config["dataset"], config["records"], config["stride_filtering_output"])
    turns_detector.run()
    logger("Turns detector completed")

    # Step 2: Estimate spatio-temporal parameters
    spatiotemporal_estimator = SpatioTemporalEstimator(
        config["dataset"], config["records"], config["stride_filtering_output"])
    spatiotemporal_estimator.find_outliers()
    logger("Spatio-temporal estimator completed")

    # Step 3: Parse kinematics files
    kinematics_processor = KinematicsProcessor(
        config["dataset"], config["records"], config["stride_filtering_output"], config["kinematics_filtering_threshold"])
    kinematics_processor.run()
    logger("Kinematics filtering completed")

    # Step 4: Generate final filtering combining previous info
    FinalFiltering(config["stride_filtering_output"]).run()

    # Step 5: Generate processed files with filtered strides
    spatiotemporal_estimator.generate_filtered_resume()
    kinematics_processor.generate_filtered_resume()

    logger("Preprocessing completed")


if __name__ == "__main__":
    main()
