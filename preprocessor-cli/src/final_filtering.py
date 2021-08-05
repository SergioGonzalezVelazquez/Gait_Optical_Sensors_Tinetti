#! /usr/bin/env python3
import pandas as pd
import numpy as np
from utils import *
import ast


class FinalFiltering:
    """Doc"""

    def __init__(self, filtering):
        self.filtering_path = filtering
        self.stride_filtering = None

    def load_stride_filtering(self):
        self.stride_filtering = pd.read_excel(self.filtering_path)

    def generate_filtering(self, row, foot):
        turns_filtering = ast.literal_eval(row['Turns_' + foot.title() + '_Strides_Filtering'])
        spatiotemporal_filtering = ast.literal_eval(row['Spatiotemporal_' + foot.title() + '_Strides_Filtering'])
        turns_and_spatiotemporal = list(set(turns_filtering) & set(spatiotemporal_filtering))
        kinematics_filtering = ast.literal_eval(row['Kinematics_' + foot.title() + '_Strides_Filtering'])

        result = list(set(turns_and_spatiotemporal + kinematics_filtering))
        result.sort()

        return result

    def run(self):
        # Load filtering file
        self.load_stride_filtering()

        # Generate new filtering column
        self.stride_filtering["Left_Strides_Filtering"] = self.stride_filtering.apply(lambda row: self.generate_filtering(row, 'LEFT'), axis = 1)
        self.stride_filtering["Right_Strides_Filtering"] = self.stride_filtering.apply(lambda row: self.generate_filtering(row, 'RIGHT'), axis = 1)

        # Update filtering file
        self.stride_filtering.to_excel(self.filtering_path, index=False)
