import datetime
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter

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

'''
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
    '''

def isValidMissingEventsStride(stride, foot, ref_sequence, first_event, last_event):
    if len(stride) < 3:
        return False

    if len(stride) == 4:
        stride_events = [i[1] for i in stride]
        missing_event = list(set(ref_sequence) - set(stride_events))[0]
        no_allowed_missed = ['EVENT_' + foot +'_FOOT_TOE_OFF', 'EVENT_' + opposite_foot(foot) +'_FOOT_TOE_OFF', ]
        if (missing_event == 'EVENT_' + foot +'_FOOT_TOE_OFF'):
            return False
        elif (missing_event == 'EVENT_' + opposite_foot(foot) +'_FOOT_TOE_OFF'):
            if ((foot + '_FOOT_TOE_OFF' in first_event) and (opposite_foot(foot) + '_FOOT_INITIAL_CONTACT' in last_event)):
                return False
    
    return True

def isValidExtraEventsStride(stride, foot):
    # Cuenta el número de veces que aparece cada evento sin tener en cuenta el primero y el último (aparecerán repetidos)
    counter = Counter([x[1] for idx, x in enumerate(stride) if idx not in [0, len(stride) - 1]])
    
    # Obtiene una lista de las extremidades para las que se repiten eventos
    foots = list(set(map(lambda  x: x.split("_")[1], [item for item, cnt in counter.items() if cnt > 1])))
    
    # Si se repiten eventos para la pierna de la zancada, no es válida
    return not foot.upper() in foots

def getEventTime(stride, event):
    events = [evt[1] for evt in stride]
    
    # Check if event is present
    event_pos = [index for index, evnt in enumerate(events) if evnt == event]
    if len(event_pos) > 0:
        event_pos = event_pos[len(event_pos) - 1]
        return stride[event_pos][2]
    else:
        return np.NaN

def checkValidTimes(stride, foot):
    maximal_stride_time = 2.25 # (Najafi et al., 2003)
    maximal_initial_ds_time = maximal_stride_time * 0.2  # (Hollman et al., 2011)
    
    initial_contact1_time = stride[0][2]
    initial_contact2_time = stride[len(stride) - 1][2]
    stride_duration = initial_contact2_time - initial_contact1_time
    
    op_foot = opposite_foot(foot)
    op_foot_toe_off = getEventTime(stride, 'EVENT_' + op_foot + '_FOOT_TOE_OFF')
    double_support1_duration = op_foot_toe_off - initial_contact1_time

    
    # Porcentaje de balanceo superior al 75%
    foot_toe_off = getEventTime(stride, 'EVENT_' + foot + '_FOOT_TOE_OFF')
    stance_duration = foot_toe_off - initial_contact1_time
    stance_percent = (stance_duration / stride_duration) * 100
    
    
    if not np.isnan(double_support1_duration):
        return not ((double_support1_duration > maximal_initial_ds_time) and (stance_percent > 75))
    
    return True

def fix_stride_events(stride_events, foot, first_event, last_event):
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
    
    # Zancadas formadas por más de 5 eventos: analízalos y  si los eventos que sobran son todos de la extremidad opuesta 
    # se mantiene la zancada. En cambio, si sobran eventos de la misma extremidad, elimina la zancada. 
    incorrect_strides = [index for index, stride in enumerate(
        stride_events) if len(stride) > 5 and not isValidExtraEventsStride(stride, foot)]

    stride_events = [event for index, event in enumerate(stride_events) if index not in incorrect_strides]
    
    # Zancadas formadas por más de 5 eventos que no han sido eliminadas: busca y elimina los eventos sobrantes
    incorrect_strides = [(index, stride) for index, stride in enumerate(
        stride_events) if len(stride) > 5]

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


        stride_events[pos] = [event for index, event in enumerate(stride_events[pos]) if index not in errors]

    # Elimina zancadas al inicio de la grabación que estén incompletas y comiencen con la extremidad izquierda
    stride_events = stride_events[(list(map(lambda stride: len(stride) < 4, stride_events))).index(False):] if foot == 'LEFT' else stride_events
            
    # Zancadas formadas por menos de 5 eventos: elimina aquellas que tengan menos de 4 eventos o a las 
    # que 
    initial_contact_ref = ref_sequence[0]
    incorrect_strides = [index for index, 
                         stride in enumerate(stride_events) if (len(stride) < 5
                         and not isValidMissingEventsStride(stride, foot, ref_sequence, first_event, last_event))
                        ]
    
    stride_events = [event for index, event in enumerate(stride_events) if index not in incorrect_strides]

    
    return stride_events

def split_stride(events, foot):
    first_event = events[0][1]
    last_event = events[len(events) - 1][1]

    # Busca el primer contacto inicial
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

    # Busca zancadas que no están formadas 5 eventos y las corrige
    is_correct = all(len(stride) == 5 for stride in events)
    if not is_correct:
        events = fix_stride_events(events, foot, first_event, last_event)
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

def write_sheet_from_dict(writer, sheet_name, data):        
    workbook = writer.book
    workbook.create_sheet(sheet_name)
    writer.sheets = {x.title: x for x in workbook.worksheets}
    worksheet = writer.sheets[sheet_name]
    
    for irow, row_name in enumerate(data.keys()):
        worksheet.cell(irow+1, 1, row_name)
        worksheet.cell(irow+1, 2, data[row_name])
    