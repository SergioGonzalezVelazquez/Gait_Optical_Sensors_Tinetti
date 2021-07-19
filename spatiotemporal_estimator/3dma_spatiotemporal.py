#! /usr/bin/env python3
import pandas as pd
import numpy as np
import json
import os
import datetime
from functools import reduce


def split_stride(events, foot):
    first_initial_contact = [tup for tup in events if (foot + '_' + 'FOOT_INITIAL_CONTACT') in tup[1]][0]
    events_filter = events[first_initial_contact[0]:]
    events_filter = list(map(lambda x: (x[0], x[1][1], x[1][2]), enumerate(events_filter)))
    
    stride_start = [tup[0] for tup in events if tup[1] == ('EVENT_' + foot + '_FOOT_INITIAL_CONTACT')]
    stride_start_idx = [(stride_start[i], stride_start[i + 1]) for i in range(len(stride_start) - 1)]
    
    return [events[s:e+1] for s,e in stride_start_idx]

def opposite_foot(foot):
    left = 'LEFT'
    right = 'RIGHT'
    if(foot.upper() == left):
        return right
    elif (foot.upper() == right):
        return left
    else :
        return ''

def temporal_parameters(stride_events, foot):
    '''
    Generate a dataframe with some temporal parameters of a particular stride
    '''
    foot = foot.lower()
    op_foot = opposite_foot(foot).lower()
    
    # Prepare dataframe
    df = pd.DataFrame([[evnt[2] for evnt in  stride] for stride in stride_events], 
                  columns=[(foot + '_initial_contact_1'),(op_foot + '_toe_off'), 
                           (op_foot+'_initial_contact'), (foot + '_toe_off'), 
                           (foot + '_initial_contact_2')])
    
    df.index.name = 'stride'
    
    # Estimate temporal parameters
    df['stride_duration'] = df[(foot + '_initial_contact_2')] - df[(foot + '_initial_contact_1')]
    df['step_duration'] = df[(op_foot + '_initial_contact')] - df[(foot + '_initial_contact_1')]
    df['stance_duration'] = df[(foot + '_toe_off')] - df[(foot + '_initial_contact_1')]
    df['swing_duration'] =  df[(foot + '_initial_contact_2')] - df[(foot + '_toe_off')]
    df['stance_percent'] = (df['stance_duration'] / df['stride_duration']) * 100
    df['swing_percent'] =  (df['swing_duration'] / df['stride_duration']) * 100
    df['double_support1_duration'] =  df[(op_foot + '_toe_off')] - df[(foot + '_initial_contact_1')]
    df['double_support2_duration'] =  df[(foot + '_toe_off')] - df[(op_foot +'_initial_contact')]
    df['double_support_percent'] =  ((df['double_support1_duration'] + 
                                      df['double_support2_duration']) / df['stride_duration']) * 100

    # Start row index from 1
    df.index = df.index + 1
    
    df.round(2)
    return df

def stride_length(start, end, foot, raw):
    pos_init = raw[foot + "_heel_PZ"][int(start * 100):int((start * 100) + 10)].mean()
    pos_end = raw[foot + "_heel_PZ"][int(end * 100):int((end * 100) + 10)].mean()
    return abs(pos_end - pos_init)

def stride_heel_height(start, end, foot, raw):
    return (raw[foot + "_heel_PY"][int(start * 100):int((end * 100) + 10)].max())

def logger(text):
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("["+ current_time + "] ", text)


# https://stackoverflow.com/questions/55246202/remove-default-formatting-in-header-when-converting-pandas-dataframe-to-excel-sh
# https://stackoverflow.com/questions/49937041/pandas-exporting-to-excel-without-format
def write_sheet_from_df(writer, sheet_name, data, index=True):        
    # Write data
    data.to_excel(writer, sheet_name=sheet_name, startrow=1, startcol=1, header=False, index=False)
    
    # Get workbook and worksheet objects
    worksheet = writer.sheets[sheet_name]
    column_list = data.columns
    
    # Write header names
    for icol, col_name in zip(range(len(column_list)+1), column_list):
        worksheet.cell(1, icol+2, col_name)
        
    if index:
        worksheet.cell(1, 1, data.index.name)
        for irow, row_name in enumerate(data.index):
            worksheet.cell(irow+2, 1, row_name)

def write_sheet_from_dict(writer, sheet_name, data):        
    workbook = writer.book
    workbook.create_sheet(sheet_name)
    writer.sheets = {x.title: x for x in workbook.worksheets}
    worksheet = writer.sheets[sheet_name]
    
    for irow, row_name in enumerate(data.keys()):
        worksheet.cell(irow+1, 1, row_name)
        worksheet.cell(irow+1, 2, data[row_name])

def write_output(outdir, overview, right_strides_df, left_strides_df):
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(os.path.join(outdir, 'gait_analysis.xlsx'), engine='openpyxl')

    # Write workseet from overview map
    write_sheet_from_dict(writer, 'Overview', overview)

    # Write each dataframe to a different worksheet.
    write_sheet_from_df(writer, 'Right_Gait_Cycle', right_strides_df)
    write_sheet_from_df(writer, 'Left_Gait_Cycle', left_strides_df)

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


def load_raw_data(path):
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
                  usecols= range(len(headers)), names=headers)
    
    return raw

def process_record(data_dir, subject_id, record_id):
    record_path = os.path.join(data_dir, subject_id, record_id)
    
    # Load events data
    events_table = pd.read_csv(os.path.join(record_path, subject_id + "_" 
        + record_id + ".events.txt"), sep="\t", header=None, names=["index", "event", "time"])
    events = [tuple(x) for x in events_table.values]

    # Load raw data
    raw = load_raw_data(os.path.join(record_path, subject_id + "_" 
        + record_id + ".raw"))

    right_events = split_stride(events, 'RIGHT')
    right_strides_df = temporal_parameters(right_events, 'RIGHT')

    left_events = split_stride(events, 'LEFT')
    left_strides_df = temporal_parameters(left_events, 'LEFT')

    # Calculate gait time
    if (left_events[0][0][2]) < (right_events[0][0][2]):
        gait_time = (left_events[-1][-1][2]) - (left_events[0][0][2])
    else:
        gait_time = (right_events[-1][-1][2]) - (right_events[0][0][2])

    # Estimate stride length
    right_strides_df['stride_length'] = right_strides_df.apply(lambda stride: 
                                                    stride_length(stride.right_initial_contact_1, 
                                                                  stride.right_initial_contact_2, 'right', raw), 
                                                    axis = 1)

    left_strides_df['stride_length'] = left_strides_df.apply(lambda stride: 
                                                    stride_length(stride.left_initial_contact_1, 
                                                                  stride.left_initial_contact_2, 'left', raw), 
                                                    axis = 1)

    # Estimate step length

    # Estimate heel height
    right_strides_df['max_heel_height'] = right_strides_df.apply(lambda stride: 
                                                    stride_heel_height(stride.right_initial_contact_1, 
                                                                  stride.right_initial_contact_2, 'right', raw), 
                                                    axis = 1)

    left_strides_df['max_heel_height'] = left_strides_df.apply(lambda stride: 
                                                    stride_heel_height(stride.left_initial_contact_1, 
                                                                  stride.left_initial_contact_2, 'left', raw), 
                                                    axis = 1)

    # Calculate overview
    right_means = right_strides_df.mean().round(2)
    left_means = left_strides_df.mean().round(2)
    overview = {
        'Total_Strides_Left': len(left_events),
        'Total_Strides_Rigth': len(right_events),
        'Gait_Time': gait_time,
        'Cadence': 0,
        'Speed': 0,
        'Right_Stance_Phase_Duration': right_means['stance_duration'],
        'Right_Stance_Percent': right_means['stance_percent'],
        'Right_Swing_Phase_Duration': right_means['swing_duration'],
        'Right_Swing_Phase_Percent': right_means['swing_percent'],
        'Left_Stance_Phase_Duration': left_means['stance_duration'],
        'Left_Stance_Percent': left_means['stance_percent'],
        'Left_Swing_Phase_Duration': left_means['swing_duration'],
        'Left_Swing_Phase_Percent': left_means['swing_percent'],
        'Double_Support_Percent': (left_means['double_support_percent'] + right_means['double_support_percent']) / 2,
        'Right_Stride_Length': right_means['stride_length'],
        'Left_Stride_Length': left_means['stride_length'],
        'Right_Step_Length': 0,
        'Left_Step_Length': 0,
        'Right_Stride_Duration': right_means['stride_duration'],
        'Left_Stride_Duration': left_means['stride_duration'],
        'Right_Step_Duration': right_means['step_duration'],
        'Left_Step_Duration': left_means['step_duration'],
        'Base_Of_Support': 0,
        'Right_Heel_Height': right_strides_df['max_heel_height'].max(),
        'Left_Heel_Height': left_strides_df['max_heel_height'].max(),
    }

    # Write output
    write_output(record_path, overview, right_strides_df, left_strides_df)

def main():
    # Read config file
    try: 
        with open("config.json", "r") as config:
            config_file=json.load(config)
            records = config_file["records"]
    except OSError as e:
        logger("File config.json not found")
        sys.exit()
    
    # Load file with anthropometry data
    anthropometry = pd.read_excel(config_file["anthropometry_data"], skipfooter=1, index_col=0)
    anthropometry

    for subject in records.keys():
        for record_id in records[subject]:
            process_record(config_file["data"], subject, record_id)
  
    

if __name__ == "__main__":
    main()
    print("hecho")
    