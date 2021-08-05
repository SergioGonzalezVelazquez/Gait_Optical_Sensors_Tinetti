#! /usr/bin/env python3
import pandas as pd
import numpy as np
import json
import os
import datetime
from functools import reduce


def logger(text):
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("[" + current_time + "] ", text)


def load_events(events_path):
    # Load events file
    events_table = pd.read_csv(events_path, sep="\t", header=None, names=[
                               "index", "event", "time"])
    events = [tuple(x) for x in events_table.values]

    # Preprocess event file: Remove events before first initial contact
    first_initial_contact = [tup for tup in events if (
        'FOOT_INITIAL_CONTACT') in tup[1]][0]
    events_filter = events[first_initial_contact[0]:]
    events_filter = list(
        map(lambda x: (x[0], x[1][1], x[1][2]), enumerate(events_filter)))
    events = events_filter

    return events


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
                      usecols=range(len(headers)), names=headers)

    return raw


def fix_stride_events(stride_events, foot):
    if foot == 'LEFT':
        ref_sequence = [
            'EVENT_LEFT_FOOT_INITIAL_CONTACT',
            'EVENT_RIGHT_FOOT_TOE_OFF',
            'EVENT_RIGHT_FOOT_INITIAL_CONTACT',
            'EVENT_LEFT_FOOT_TOE_OFF',
            'EVENT_LEFT_FOOT_INITIAL_CONTACT'
        ]
    else:
         ref_sequence = [
            'EVENT_RIGHT_FOOT_INITIAL_CONTACT',
            'EVENT_LEFT_FOOT_TOE_OFF',
            'EVENT_LEFT_FOOT_INITIAL_CONTACT',
            'EVENT_RIGHT_FOOT_TOE_OFF',
            'EVENT_RIGHT_FOOT_INITIAL_CONTACT'
        ]       

    incorrect_strides = [(index, stride) for index, stride in enumerate(
        stride_events) if len(stride) > len(ref_sequence)]

    for inc_stride in incorrect_strides:
        pos = inc_stride[0]
        stride = inc_stride[1]

        seq_index = 0
        errors = []
        for event_index, event in enumerate(stride):
            if event[1] == ref_sequence[seq_index]:
                seq_index += 1
            else:
                errors.append(event_index)

        for error in errors:
            stride_events[pos] = [event for index, event in enumerate(stride_events[pos]) if index not in errors]

    return stride_events


def split_stride(events, foot):
    first_initial_contact = [tup for tup in events if (
        foot + '_' + 'FOOT_INITIAL_CONTACT') in tup[1]][0]
    events_filter = events[first_initial_contact[0]:]
    events_filter = list(
        map(lambda x: (x[0], x[1][1], x[1][2]), enumerate(events_filter)))

    stride_start = [tup[0] for tup in events if tup[1]
                    == ('EVENT_' + foot + '_FOOT_INITIAL_CONTACT')]
    stride_start_idx = [(stride_start[i], stride_start[i + 1])
                        for i in range(len(stride_start) - 1)]

    events = [events[s:e+1] for s, e in stride_start_idx]

    # Comprueba que todas las zancadas están formadas por 5 eventos
    is_correct = all(len(stride) <= 5 for stride in events)
    if not is_correct:
        events = fix_stride_events(events, foot)

    return events


def opposite_foot(foot):
    left = 'LEFT'
    right = 'RIGHT'
    if(foot.upper() == left):
        return right
    elif (foot.upper() == right):
        return left
    else:
        return ''


def temporal_parameters(stride_events, foot):
    foot = foot.lower()
    op_foot = opposite_foot(foot).lower()

    # Prepare dataframe
    df = pd.DataFrame([[evnt[2] for evnt in stride] for stride in stride_events],
                      columns=[(foot + '_initial_contact_1'), (op_foot + '_toe_off'),
                               (op_foot+'_initial_contact'), (foot + '_toe_off'),
                               (foot + '_initial_contact_2')])

    df.index.name = 'stride'

    # Estimate temporal parameters
    df['stride_duration'] = df[(foot + '_initial_contact_2')] - \
        df[(foot + '_initial_contact_1')]
    df['step_duration'] = df[(op_foot + '_initial_contact')] - \
        df[(foot + '_initial_contact_1')]
    df['stance_duration'] = df[(foot + '_toe_off')] - \
        df[(foot + '_initial_contact_1')]
    df['swing_duration'] = df[(foot + '_initial_contact_2')
                              ] - df[(foot + '_toe_off')]
    df['stance_percent'] = (df['stance_duration'] /
                            df['stride_duration']) * 100
    df['swing_percent'] = (df['swing_duration'] / df['stride_duration']) * 100
    df['double_support1_duration'] = df[(
        op_foot + '_toe_off')] - df[(foot + '_initial_contact_1')]
    df['double_support2_duration'] = df[(
        foot + '_toe_off')] - df[(op_foot + '_initial_contact')]
    df['double_support_percent'] = ((df['double_support1_duration'] +
                                     df['double_support2_duration']) / df['stride_duration']) * 100

    # Start row index from 1
    df.index = df.index + 1

    df.round(2)
    return df

# https://stackoverflow.com/questions/55246202/remove-default-formatting-in-header-when-converting-pandas-dataframe-to-excel-sh
# https://stackoverflow.com/questions/49937041/pandas-exporting-to-excel-without-format


def write_sheet_from_df(writer, sheet_name, data, index=True):
    # Write data
    data.to_excel(writer, sheet_name=sheet_name, startrow=1,
                  startcol=1, header=False, index=False)

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


def write_output(output_path, right_strides_spatiotemporal, left_strides_spatiotemporal):
    right_means = right_strides_spatiotemporal.mean().round(2)
    left_means = right_strides_spatiotemporal.mean().round(2)

    overview = {
        'Total_Strides_Left': len(left_strides_spatiotemporal),
        'Total_Strides_Rigth': len(right_strides_spatiotemporal),
        # 'Gait_Time': gait_time,
        # 'Cadence': 0,
        # 'Speed': 0,
        'Right_Stance_Phase_Duration': right_means['stance_duration'],
        'Right_Stance_Percent': right_means['stance_percent'],
        'Right_Swing_Phase_Duration': right_means['swing_duration'],
        'Right_Swing_Phase_Percent': right_means['swing_percent'],
        'Left_Stance_Phase_Duration': left_means['stance_duration'],
        'Left_Stance_Percent': left_means['stance_percent'],
        'Left_Swing_Phase_Duration': left_means['swing_duration'],
        'Left_Swing_Phase_Percent': left_means['swing_percent'],
        # 'Double_Support_Percent': (left_means['ds_percent'] + right_means['ds_percent']) / 2,
        # 'Right_Stride_Length': right_means['stride_length'],
        # 'Left_Stride_Length': left_means['stride_length'],
        # 'Right_Step_Length': 0,
        # 'Left_Step_Length': 0,
        'Right_Stride_Duration': right_means['stride_duration'],
        'Left_Stride_Duration': left_means['stride_duration'],
        'Right_Step_Duration': right_means['step_duration'],
        'Left_Step_Duration': left_means['step_duration'],
        # 'Base_Of_Support': 0,
        # 'Right_Heel_Height': right_strides_df['max_heel_height'].max(),
        # 'Left_Heel_Height': left_strides_df['max_heel_height'].max(),
    }

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(output_path, engine='openpyxl')

    # Write workseet from overview map
    write_sheet_from_dict(writer, 'Overview', overview)

    # Write each dataframe to a different worksheet.
    write_sheet_from_df(writer, 'Right_Gait_Cycle',
                        right_strides_spatiotemporal)
    write_sheet_from_df(writer, 'Left_Gait_Cycle', left_strides_spatiotemporal)

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


def process_record(dataset, subject_id, record_id):
    data_path = os.path.join(dataset, subject_id, record_id)

    # Load events file
    events_path = os.path.join(
        data_path, (subject_id + "_" + record_id + ".events.txt"))
    events = load_events(events_path)

    # Split events by strides
    right_events = split_stride(events, 'RIGHT')
    left_events = split_stride(events, 'LEFT')

    # Load raw data
    raw_path = os.path.join(data_path, (subject_id + "_" + record_id + ".raw"))
    raw = load_raw_data(raw_path)

    # Estimate temporal parameters
    try:
        right_strides_temporal_df = temporal_parameters(right_events, 'RIGHT')
        left_strides_temporal_df = temporal_parameters(left_events, 'LEFT')
    except:
        print("error")
        return -1

    # Generate output
    output_path = os.path.join(dataset, subject_id, record_id,
                               (subject_id + "_" + record_id + ".spatiotemporal.xlsx"))
    write_output(output_path, right_strides_temporal_df,
                 left_strides_temporal_df)

    return 0


def main():
    # Read config file
    try:
        with open("config.json", "r") as config:
            config_file = json.load(config)
            records = config_file["records"]
    except OSError as e:
        logger("File config.json not found")
        sys.exit()

    # Load file with anthropometry data
    anthropometry = pd.read_excel(
        config_file["anthropometry_data"], skipfooter=1, index_col=0)
    anthropometry
    errors = []
    for subject in records.keys():
        logger("Process " + subject)
        for record_id in records[subject]:
            logger("Record " + record_id)
            result = process_record(config_file["data"], subject, record_id)
            if result < 0:
                errors.append(subject + "_" + record_id)

    print("errors: " + str(len(errors)))
    print(errors)


if __name__ == "__main__":
    main()
