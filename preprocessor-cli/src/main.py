#! /usr/bin/env python3
from utils import *
from turns_detector import TurnsDetector
from spatiotemporal_estimator import SpatioTemporalEstimator
import json
import sys

def load_config():
    # Read config file
    try:
        with open("config.json", "r") as config:
            config_file = json.load(config)
            return config_file
    except OSError as e:
        logger("File config.json not found", msgType=LoggerType.FAIL)
        sys.exit()

def main():
    logger("Starting preprocessing")
    config = load_config()

    # Step 1: Run turns-detector
    #turns_detector = TurnsDetector(config["dataset"], config["records"], config["stride_filtering_output"])
    #turns_detector.run()
    logger("Turns detector completed")

    # Step 2: Estimate spatio-temporal parameters
    spatiotemporal_estimator = SpatioTemporalEstimator(config["dataset"], config["records"], config["stride_filtering_output"])
    spatiotemporal_estimator.run()
    logger("Spatio-temporal estimator completed")

    # Step 3: Parse kinematics files

    logger("Preprocessing completed")

if __name__ == "__main__":
    main()