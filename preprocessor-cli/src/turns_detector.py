'''TODO'''
import os
import pandas as pd
import numpy as np
from peakdetect import peakdetect
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from utils import *


class TurnsDetector:
    '''TODO'''

    def __init__(self, dataset_path, records, output, enable_logger=True):
        self.dataset = dataset_path
        self.records = records
        self.output = output
        self.enable_logger = enable_logger
        self.turn_list = []
        self.errors = []

    def log(self, text, msgType=LoggerType.OKGREEN):
        if self.enable_logger:
            logger(text, msgType=msgType)

    def process_record(self, dataset, subject_id, record_id, output_file):
        '''TODO'''
        data_path = os.path.join(dataset, subject_id, record_id)

        # Load events file
        events_path = os.path.join(
            data_path, (subject_id + "_" + record_id + ".events.txt"))
        events = load_events(events_path)

        # Split events by strides
        right_events = split_stride(events, 'RIGHT')
        left_events = split_stride(events, 'LEFT')

        # Load CDG file
        cdg_path = os.path.join(data_path, "biomechanics/COG.csv")
        cdg = self.load_cdg(cdg_path)
        cdg = cdg[(cdg["Tiempo"] >= events[0][2])]

        # Find turns
        turns, left_filter, right_filter = self.find_turns(
            cdg, events, left_events, right_events, output_file, subject_id, record_id)
        return turns, left_filter, right_filter

    def plot_charts(self, time, cdg_z, cdg_z_hat, peaks_times, peaks_vals, events, output_path, filename):
        '''TODO'''
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)

        labels = []
        for event in events:
            color = "red" if "LEFT" in event[1] else "blue"
            linestyle = "solid" if "INITIAL_CONTACT" in event[1] else "dashed"
            if not event[1] in labels:
                plt.axvline(x=event[2], c=color, alpha=0.2,
                            label=event[1], ls=linestyle)
                labels.append(event[1])
            else:
                plt.axvline(x=event[2], c=color, alpha=0.2, ls=linestyle)

        plt.xlim(0, 30)
        plt.plot(time, cdg_z, c="black")
        plt.plot(peaks_times, peaks_vals, 'bo', ms=7,)
        plt.ylabel('cm')
        plt.xlabel('Tiempo')

        plt.subplot(1, 2, 2)

        labels = []
        for event in events:
            color = "red" if "LEFT" in event[1] else "blue"
            linestyle = "solid" if "INITIAL_CONTACT" in event[1] else "dashed"
            if not event[1] in labels:
                plt.axvline(x=event[2], c=color, alpha=0.2,
                            label=event[1], ls=linestyle)
                labels.append(event[1])
            else:
                plt.axvline(x=event[2], c=color, alpha=0.2, ls=linestyle)

        plt.xlim(0, 30)
        plt.plot(time, cdg_z_hat, c="black")
        plt.plot(peaks_times, peaks_vals, 'bo', ms=7,)
        plt.ylabel('cm')
        plt.xlabel('Tiempo')

        save_fig(filename, os.path.join(os.path.dirname(
            output_path), "turns_stride_filtering"))
        plt.close('all')

    def find_turns(self, cdg, events, left_events, right_events, out_dir, subject, record):
        '''TODO'''
        time = cdg["Tiempo"].to_numpy()
        cdg_z = cdg["CDG Z"].to_numpy()

        # Smooth data
        cdg_z_hat = savgol_filter(cdg_z, 901, 3)

        peaks = peakdetect(cdg_z_hat, time, lookahead=20)

        higher_peaks_times = []
        higher_peaks_vals = []
        lower_peaks_times = []
        lower_peaks_vals = []

        # Higher peaks
        if (peaks[0]):
            higher_peaks_times = np.array(peaks[0])[:, 0].tolist()
            higher_peaks_vals = np.array(peaks[0])[:, 1].tolist()

        # Lower peaks
        if (peaks[1]):
            lower_peaks_times = np.array(peaks[1])[:, 0].tolist()
            lower_peaks_vals = np.array(peaks[1])[:, 1].tolist()

        peaks_times = higher_peaks_times + lower_peaks_times
        peaks_vals = higher_peaks_vals + lower_peaks_vals

        left_filter = []
        right_filter = []
        for peak_time in peaks_times:
            # Coge el evento más cercano anterior al pico
            pre_peak_events = [
                event for event in events if event[2] <= peak_time]
            pre_peak_event = pre_peak_events[-1]

            # Coge el evento más cercano posterior al pico
            post_peak_events = [
                event for event in events if event[2] >= peak_time]
            if post_peak_events:
                post_peak_event = post_peak_events[0]

                # Ver en qué zacandas coincide el evento más cercano anterior al pico (izquierda)
                pre_left_filter = [index for index, event in enumerate(
                    left_events) if pre_peak_event in event]
                left_filter = left_filter + pre_left_filter

                # Ver en qué zacandas coincide el evento más cercano anterior al pico (derecha)
                pre_right_filter = [index for index, event in enumerate(
                    right_events) if pre_peak_event in event]
                right_filter = right_filter + pre_right_filter

                # Ver en qué zacandas coincide el evento más cercano posterior al pico (izquierda)
                post_left_filter = [index for index, event in enumerate(
                    left_events) if post_peak_event in event]
                left_filter = left_filter + post_left_filter

                # Ver en qué zacandas coincide el evento más cercano posterior al pico (derecha)
                post_right_filter = [index for index, event in enumerate(
                    right_events) if post_peak_event in event]
                right_filter = right_filter + post_right_filter

                left_filter = list(set(left_filter))
                right_filter = list(set(right_filter))
                
        left_filter = [x + 1 for x in left_filter]
        right_filter = [x + 1 for x in right_filter]
        turns = len(higher_peaks_vals) + len(lower_peaks_vals)
        #self.plot_charts(time, cdg_z, cdg_z_hat, peaks_times,
        #                peaks_vals, events, out_dir, subject + "_" + record)

        return turns, left_filter, right_filter

    def load_cdg(self, cdg_path):
        '''TODO'''
        cdg = pd.read_csv(cdg_path, sep=";", decimal=",")

        # FIX TO BUG
        if len(cdg.columns) > 4:
            cols_to_drop = []
            for column in filter(lambda col: col != "Tiempo", cdg.columns):
                if (cdg['Tiempo'].equals(cdg[column])):
                    cols_to_drop.append(column)
                else:
                    # Rename column
                    cdg.rename(
                        columns={column: column.split("_")[0]}, inplace=True)

            cdg.drop(cols_to_drop, axis=1, inplace=True)

            # Overwrite file
            cdg.to_csv(cdg_path, sep=";", decimal=",", index=False)

        cdg.drop(cdg[cdg["CDG Z"] > 600].index, inplace=True)
        cdg.drop(cdg[cdg["CDG Z"] < -600].index, inplace=True)

        return cdg

    def run(self):
        '''TODO'''
        self.log("[Turns Detector] Start", msgType=LoggerType.OKBLUE)

        for subject in self.records.keys():
            self.log("[Turns Detector] Process " +
                   subject, msgType=LoggerType.OKBLUE)
            for record_id in self.records[subject]:
                self.log("[Turns Detector] Record " +
                       record_id, msgType=LoggerType.OKBLUE)
                turns, left_filter, right_filter = self.process_record(
                    self.dataset, subject, record_id, self.output)
                self.turn_list.append(
                    (subject, record_id, turns, left_filter, right_filter))

        turns_table = pd.DataFrame(self.turn_list, columns=[
                                   "Subject", "Record", "Turns_Count", "Turns_Left_Strides_Filtering", "Turns_Right_Strides_Filtering"])
        turns_table.to_excel(self.output, index=False)
        self.log("[Turns Detector] Errors count " +
               str(len(self.errors)), msgType=LoggerType.OKBLUE)
