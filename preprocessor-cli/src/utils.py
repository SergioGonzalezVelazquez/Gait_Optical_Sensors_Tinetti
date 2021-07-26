import datetime
import pandas as pd
import os
import matplotlib.pyplot as plt

class LoggerType:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def logger(text, msgType=LoggerType.OKGREEN):
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    
    if msgType == LoggerType.WARNING:
        print(LoggerType.WARNING + "[" + current_time + "] Warning: " + text + LoggerType.ENDC)
    elif msgType == LoggerType.FAIL:
        print(LoggerType.FAIL + "[" + current_time + "] Error: " + text + LoggerType.ENDC)
    else:
        print(msgType + "[" + current_time + "] " + text + LoggerType.ENDC)


def save_fig(fig_id, out_dir, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(out_dir, fig_id + "." + fig_extension)

    # Make directory if not exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution, pad_inches=0)


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


def opposite_foot(foot):
    left = 'LEFT'
    right = 'RIGHT'
    if(foot.upper() == left):
        return right
    elif (foot.upper() == right):
        return left
    else:
        return ''


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