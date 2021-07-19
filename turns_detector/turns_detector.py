import pandas as pd
import numpy as np
import json
import os
from peakdetect import peakdetect
from tabulate import tabulate
from scipy.signal import savgol_filter
import datetime
import matplotlib.pyplot as plt

def save_fig(fig_id, out_dir, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(out_dir, fig_id + "." + fig_extension)

    # Make directory if not exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution, pad_inches=0)


def plot_charts(time, cdg_z, cdg_z_hat, peaks_times, peaks_vals, events, output_path, filename):
    plt.figure(figsize=(15,5))
    plt.subplot(1, 2, 1)

    labels = []
    for event in events:
        color = "red" if "LEFT" in event[1] else "blue"
        linestyle = "solid" if "INITIAL_CONTACT" in event[1] else "dashed"
        if not event[1] in labels:
            plt.axvline(x=event[2], c=color, alpha=0.2, label = event[1], ls=linestyle)
            labels.append(event[1])
        else:
            plt.axvline(x=event[2], c=color, alpha=0.2, ls=linestyle)

    plt.xlim(0,30)
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
            plt.axvline(x=event[2], c=color, alpha=0.2, label = event[1], ls=linestyle)
            labels.append(event[1])
        else:
            plt.axvline(x=event[2], c=color, alpha=0.2, ls=linestyle)

    plt.xlim(0,30)
    plt.plot(time, cdg_z_hat, c="black")
    plt.plot(peaks_times, peaks_vals, 'bo', ms=7,)
    plt.ylabel('cm')
    plt.xlabel('Tiempo')

    save_fig(filename, os.path.join(os.path.dirname(output_path), "turns_stride_filtering"))
    plt.close('all')

def logger(text):
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("["+ current_time + "] ", text)

def opposite_foot(foot):
    left = 'LEFT'
    right = 'RIGHT'
    if(foot.upper() == left):
        return right
    elif (foot.upper() == right):
        return left
    else :
        return ''

def split_stride(events, foot):
    first_initial_contact = [tup for tup in events if (foot + '_' + 'FOOT_INITIAL_CONTACT') in tup[1]][0]
    events_filter = events[first_initial_contact[0]:]
    events_filter = list(map(lambda x: (x[0], x[1][1], x[1][2]), enumerate(events_filter)))
    
    stride_start = [tup[0] for tup in events if tup[1] == ('EVENT_' + foot + '_FOOT_INITIAL_CONTACT')]
    stride_start_idx = [(stride_start[i], stride_start[i + 1]) for i in range(len(stride_start) - 1)]
    
    return [events[s:e+1] for s,e in stride_start_idx]

def load_events(events_path):
    # Load events file
    events_table = pd.read_csv(events_path , sep="\t", header=None, names=["index", "event", "time"])
    events = [tuple(x) for x in events_table.values]

    # Preprocess event file: Remove events before first initial contact
    first_initial_contact = [tup for tup in events if ('FOOT_INITIAL_CONTACT') in tup[1]][0]
    events_filter = events[first_initial_contact[0]:]
    events_filter = list(map(lambda x: (x[0], x[1][1], x[1][2]), enumerate(events_filter)))
    events = events_filter

    return events

def find_turns(cdg, events, left_events, right_events, out_dir, subject, record):
    time = cdg["Tiempo"].to_numpy()
    cdg_z = cdg["CDG Z"].to_numpy()

    # Smooth data
    cdg_z_hat = savgol_filter(cdg_z, 901, 3) # window size 51, polynomial order 3

    peaks = peakdetect(cdg_z_hat, time, lookahead=20)

    higher_peaks_times = []
    higher_peaks_vals = []
    lower_peaks_times = []
    lower_peaks_vals = []

    # Higher peaks
    if (peaks[0]):
        higher_peaks_times = np.array(peaks[0])[:,0].tolist()
        higher_peaks_vals =  np.array(peaks[0])[:,1].tolist()
        
    # Lower peaks
    if (peaks[1]):
        lower_peaks_times = np.array(peaks[1])[:,0].tolist()
        lower_peaks_vals =  np.array(peaks[1])[:,1].tolist()

    peaks_times =  higher_peaks_times + lower_peaks_times
    peaks_vals = higher_peaks_vals + lower_peaks_vals

    left_filter = []
    right_filter = []
    for peak_time in peaks_times:
        # Coge el evento más cercano anterior al pico
        pre_peak_events = [event for event in events if event[2] <= peak_time]
        pre_peak_event = pre_peak_events[-1]

        # Coge el evento más cercano posterior al pico
        post_peak_events = [event for event in events if event[2] >= peak_time]
        if post_peak_events:
            post_peak_event = post_peak_events[0]
        
            # Ver en qué zacandas coincide el evento más cercano anterior al pico (izquierda)
            pre_left_filter = [index for index, event in enumerate(left_events) if pre_peak_event in event]
            left_filter = left_filter + pre_left_filter
            
            # Ver en qué zacandas coincide el evento más cercano anterior al pico (derecha)
            pre_right_filter = [index for index, event in enumerate(right_events) if pre_peak_event in event]
            right_filter = right_filter + pre_right_filter
            
            # Ver en qué zacandas coincide el evento más cercano posterior al pico (izquierda)
            post_left_filter = [index for index, event in enumerate(left_events) if post_peak_event in event]
            left_filter = left_filter + post_left_filter
            
            # Ver en qué zacandas coincide el evento más cercano posterior al pico (derecha)
            post_right_filter = [index for index, event in enumerate(right_events) if post_peak_event in event]
            right_filter = right_filter + post_right_filter
            
            left_filter = list(set(left_filter))
            right_filter = list(set(right_filter))

    turns = len(higher_peaks_vals) + len(lower_peaks_vals)
    plot_charts(time, cdg_z, cdg_z_hat, peaks_times, peaks_vals, events, out_dir, subject + "_" + record)

    return turns, left_filter, right_filter

def load_cdg(cdg_path):
    cdg = pd.read_csv(cdg_path, sep=";", decimal=",")

    cdg.drop(cdg[cdg["CDG Z"] > 600].index, inplace=True)
    cdg.drop(cdg[cdg["CDG Z"] < -600].index, inplace=True)

    # FIX TO BUG
    if len(cdg.columns) > 4:
        cols_to_drop = []
        for column in filter(lambda col: col != "Tiempo", cdg.columns):
            if (cdg['Tiempo'].equals(cdg[column])):
                cols_to_drop.append(column)
            else:
                # Rename column
                cdg.rename(columns={column: column.split("_")[0]}, inplace=True)
    
        cdg.drop(cols_to_drop, axis=1, inplace=True)
    
        # Overwrite file
        cdg.to_csv(cdg_path, sep=";", decimal=",", index=False)

    return cdg

def process_record(dataset, subject_id, record_id, output_file):
    data_path = os.path.join(dataset, subject_id, record_id)
    
    # Load events file
    events_path = os.path.join(data_path, (subject_id + "_" + record_id + ".events.txt"))
    events = load_events(events_path)

    # Split events by strides
    right_events = split_stride(events, 'RIGHT')
    left_events = split_stride(events, 'LEFT')

    # Load CDG file
    cdg_path = os.path.join(data_path, "biomechanics/CDG.csv")
    cdg = load_cdg(cdg_path)
    cdg = cdg[(cdg["Tiempo"] >= events[0][2])]


    # Find turns 
    turns, left_filter, right_filter = find_turns(cdg, events, left_events, right_events, output_file, subject_id, record_id)
    return turns, left_filter, right_filter


def main():
    # Read config file
    try: 
        with open("config.json", "r") as config:
            config_file=json.load(config)
            output_file = config_file["output"]
            records = config_file["records"]
    except OSError as e:
        logger("File config.json not found")
        sys.exit()

    turn_list = []
    errors = []
    for subject in records.keys():
        logger("Process " + subject)
        for record_id in records[subject]:

            turns, left_filter, right_filter = process_record(config_file["data"], subject, record_id, output_file)
            # Add one to start in 1
            left_filter = [x+1 for x in left_filter]
            right_filter = [x+1 for x in right_filter]
            turn_list.append((subject, record_id, turns, left_filter, right_filter))
    
    turns_table = pd.DataFrame(turn_list, columns=["Subject", "Record", "Turns", "Left Strides", "Right Strides"])
    logger("SUMMARY:")
    print(tabulate(turns_table, headers='keys', tablefmt='psql'))
    turns_table.to_excel(output_file, index=False)
    logger("Errors count: " + str(len(errors)))

if __name__ == "__main__":
    main()