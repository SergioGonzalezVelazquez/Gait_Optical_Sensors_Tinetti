#! /usr/bin/env python3
import pyautogui
import psutil
import time
import sys
import json
import os
import shutil
import datetime


STT_REPORTS = "C:/Users/WeCareLab/Documents/STT/3DMA/Reports"

def process_exists(process_name):
    return process_name in (p.name() for p in psutil.process_iter())

def countdown(seconds, text=''):
    while seconds:
        timer = '%s %d' % (text, seconds)
        print(timer, end="\r")
        time.sleep(1)
        seconds -=1
    logger("Starting")


def load_user(user_id):
    # Click on "Select user"
    pyautogui.click(41, 89)

    # Click on advanced search
    pyautogui.moveTo(882, 98, duration=1)
    pyautogui.click(882, 98)

    # Click on search
    pyautogui.click(223, 103)

    # Type username
    pyautogui.typewrite(user_id)

    # Select user
    pyautogui.moveTo(123, 173, duration=1)
    pyautogui.click(123, 173)

    # Confirm
    pyautogui.moveTo(1680, 986, duration=1)
    pyautogui.click(1680, 986)

    time.sleep(2)


def open_record(record):
    # Click on "Open record"
    pyautogui.click(94, 94)

    # Click on search
    pyautogui.moveTo(1177, 180, duration=1)
    pyautogui.click(1177, 180,)

    # Type record name
    pyautogui.typewrite(record)

    # Select record
    pyautogui.moveTo(1036, 248, duration=1)
    pyautogui.click(1036, 248)

    # Confirm
    pyautogui.moveTo(1665, 977, duration=1)
    pyautogui.click(1665, 977)

    time.sleep(10)

def handle_file_explorer(path, filename=None):
    if filename:
        # Write filename
        pyautogui.keyDown('del')
        time.sleep(.4)
        pyautogui.keyUp('del')
        pyautogui.typewrite(filename)
        time.sleep(.2)

    # File explorer dialog: focus address bar and type directory name
    pyautogui.hotkey('alt', 'd')
    time.sleep(.2)
    pyautogui.typewrite(path)
    time.sleep(.2)
    pyautogui.press('enter')
    time.sleep(1)
    pyautogui.press('enter')
    pyautogui.press('enter')
    time.sleep(1)
    pyautogui.press('enter')

def export_3mc(path, filename):
    # Click on export data
    pyautogui.moveTo(324, 80, duration=1)
    pyautogui.click(324, 80,)

    # Click on export 3MC
    pyautogui.moveTo(324, 149, duration=1)
    pyautogui.click(324, 149,)
    time.sleep(2)
    
    # File explorer dialog
    handle_file_explorer(path, filename)

    time.sleep(.5)
    

def export_3dt(path, filename):
    # Click on export data
    pyautogui.moveTo(324, 80, duration=1)
    pyautogui.click(324, 80,)

    # Click on export 3DT
    pyautogui.moveTo(324, 263, duration=1)
    pyautogui.click(324, 263,)
    time.sleep(2)
    
    # File explorer dialog
    handle_file_explorer(path, filename)

    time.sleep(.5)
    

def export_trc(path, filename):
    # Click on export data
    pyautogui.moveTo(324, 80, duration=1)
    pyautogui.click(324, 80,)

    # Click on export TRC
    pyautogui.moveTo(324, 331, duration=1)
    pyautogui.click(324, 331,)
    time.sleep(2)

    # Open on file explorer
    pyautogui.moveTo(1144, 650, duration=1)
    pyautogui.click(1144, 650,)
    time.sleep(.3)   
    
    # File explorer dialog
    handle_file_explorer(path, filename)

    # Click on accept
    pyautogui.moveTo(889, 679, duration=1)
    pyautogui.click(889, 679,)   

    time.sleep(.5)


def export_c3d(path, filename):
    # Click on export data
    pyautogui.moveTo(324, 80, duration=1)
    pyautogui.click(324, 80,)

    # Click on export C3D
    pyautogui.moveTo(377, 351, duration=1)
    pyautogui.click(377, 351,)
    time.sleep(1)

    # Click on accept
    pyautogui.moveTo(923, 692, duration=1)
    pyautogui.click(923, 692,)
    time.sleep(.5)

    # File explorer dialog
    handle_file_explorer(path, filename)

    time.sleep(.5)


def export_raw(path, filename):
    # Click on export data
    pyautogui.moveTo(324, 80, duration=1)
    pyautogui.click(324, 80,)

    # Click on export RAW
    pyautogui.moveTo(368, 399, duration=1)
    pyautogui.click(368, 399,)
    time.sleep(2)
    
    # File explorer dialog
    handle_file_explorer(path, filename + '.raw')

    time.sleep(.5)


def export_events(path, filename):
    # Click on export data
    pyautogui.moveTo(324, 80, duration=1)
    pyautogui.click(324, 80,)

    # Click on export events
    pyautogui.moveTo(368, 452, duration=1)
    pyautogui.click(368, 452,)
    time.sleep(2)
    
    # File explorer dialog
    handle_file_explorer(path, filename + '.events')

    time.sleep(.5)


def export_doc(path, filename, subject):
    doc_path = os.path.join(STT_REPORTS, subject + '_' + filename + '_1.docx')
    if not os.path.isfile(doc_path):
        # Click on generate report
        pyautogui.moveTo(1132, 95, duration=1)
        pyautogui.click(1132, 95,)

        # Select walking
        pyautogui.click(1067, 445,)
        time.sleep(.2)
        pyautogui.click(1045, 464,)
        time.sleep(.2)
        pyautogui.moveTo(973, 593, duration=1)
        pyautogui.click(973, 593,)

        # Wait for doc
        time.sleep(15)
        while True:
            
            # Get active windows
            actwd1 = pyautogui.getActiveWindow()
            if (os.path.isfile(doc_path)) and ("Office" in actwd1.title or "Word" in actwd1.title):
                time.sleep(10)
                # Close Office licence windows
                if "Office" in actwd1.title:
                    pyautogui.moveTo(1425, 232, duration=1)
                    pyautogui.click(1425, 232,)
                    time.sleep(1)

                pyautogui.moveTo(1887, 11, duration=1)
                pyautogui.click(1887, 11,)
                break
            else:
                time.sleep(10)
    
    time.sleep(5)

    # Copy report to record directory
    shutil.copy2(doc_path, (path + '/' + filename + '.docx'))
    
def logger(text):
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("["+ current_time + "] ", text)

def export_real_video(path, filename):
    # Click on export data
    pyautogui.moveTo(324, 80, duration=1)
    pyautogui.click(324, 80,)
    time.sleep(.2)

    # Click on export
    pyautogui.moveTo(337, 584, duration=1)
    pyautogui.click(337, 584,)

    time.sleep(.4)
    
    # File explorer dialog
    handle_file_explorer(path, filename)

    # wait for video generation
    time.sleep(15)
    video_path = os.path.join(path, filename + '.mp4')
    while True:
        # Try to minimize Clinical 3DMA
        pyautogui.moveTo(1804, 9, duration=1)
        pyautogui.click(1804, 9,)

        # Get active windows
        actwd1 = pyautogui.getActiveWindow()
 
        if (os.path.isfile(video_path) and (not actwd1.title.startswith ('Clinical 3DMA'))):
            activate_3DMA()
            break
        else:
            time.sleep(15)

    time.sleep(.5)


def export_3D_video(path, filename):
    # Click on export data
    pyautogui.moveTo(324, 80, duration=1)
    pyautogui.click(324, 80,)
    time.sleep(.2)

    # Click on export 3D video
    pyautogui.moveTo(324, 536, duration=1)
    pyautogui.click(324, 536,)

    time.sleep(.4)
    
    # File explorer dialog
    handle_file_explorer(path, filename)

    # wait for video generation
    time.sleep(40)
    video_path = os.path.join(path, filename + '.mp4')
    while True:
        # Check if active windows is Clinical 3DMA
        actwd1 = pyautogui.getActiveWindow()

        if actwd1.title == 'Exportando...':
            time.sleep(40)
        else:
            break

    time.sleep(.5)



def export_strides(path):
    # Click on select stride
    pyautogui.moveTo(1076, 94, duration=1)
    pyautogui.click(1076, 94,)

    time.sleep(3)

    # Click on export strides
    pyautogui.moveTo(1361, 276, duration=1)
    pyautogui.click(1361, 276,)
    time.sleep(2)
    
    # File explorer dialog
    handle_file_explorer(path)

    # click on OK
    pyautogui.moveTo(894, 809, duration=1)
    pyautogui.click(894, 809,)

    time.sleep(.5)


def export_biomechanics(path):
    # Click on export biomechanics
    pyautogui.moveTo(1120, 713, duration=1)
    pyautogui.click(1120, 713,)

    time.sleep(2)
    
    # File explorer dialog
    handle_file_explorer(path)

    time.sleep(3)


def check_if_saved(filename):
    if os.path.isfile(filename):
        logger(filename + ' saved')
    else:
        logger(filename + " not saved")
        sys.exit()


def export_data(subject, recs, formats, output_data_dir, video_dir):
    logger(subject)
    subj_out_dir = os.path.join(output_data_dir, subject)

    # Create output dir for this subject data
    if not os.path.exists(subj_out_dir):
        os.makedirs(subj_out_dir)

    # Create output dir for this subject videos
    subj_video_dir = os.path.join(video_dir, subject)
    if not os.path.exists(subj_video_dir):
        os.makedirs(subj_video_dir)

    load_user(subject)
    for rec in recs:
        # Create output dir for this record
        if rec.startswith(subject):
            rec_out_dir = os.path.join(subj_out_dir, rec.split(subject + "_")[1])
        else:
            rec_out_dir = os.path.join(subj_out_dir, rec)
        
        if not os.path.exists(rec_out_dir):
            os.makedirs(rec_out_dir)

        open_record(rec)

        # Export biomechanics
        if "biomechanics" in formats:
            biomechanics_out_dir = os.path.join(rec_out_dir, "biomechanics")
            os.makedirs(biomechanics_out_dir)
            export_biomechanics(biomechanics_out_dir)
            filename = os.path.join(biomechanics_out_dir,  "CDG.csv")
            check_if_saved(filename)

        # Export 3MC format
        if "3mc" in formats:
            export_3mc(rec_out_dir, rec)
            filename = os.path.join(rec_out_dir, rec + ".3mc")
            check_if_saved(filename)
            
        # Export 3DT format
        if "3dt" in formats:
            export_3dt(rec_out_dir, rec)
            filename = os.path.join(rec_out_dir, rec + ".3dt")
            check_if_saved(filename)

        # Export TRC format
        if "trc" in formats:
            export_trc(rec_out_dir, rec)
            filename = os.path.join(rec_out_dir, rec + ".trc")
            check_if_saved(filename)

        # Export raw format
        if "raw" in formats:
            export_raw(rec_out_dir, rec)
            filename = os.path.join(rec_out_dir, rec + ".raw")
            check_if_saved(filename)
    
        # Export 3DT format
        if "events" in formats:
            export_events(rec_out_dir, rec)
            filename = os.path.join(rec_out_dir, rec + ".events.txt")
            check_if_saved(filename)

        # Export strides
        if "strides" in formats:
            strides_out_dir = os.path.join(rec_out_dir, "strides")
            os.makedirs(strides_out_dir)
            export_strides(strides_out_dir)
            filename = os.path.join(strides_out_dir, "Abducción aducción de cadera_Derecha.csv")
            check_if_saved(filename)

        # Export C3D format
        if "c3d" in formats:
            export_c3d(rec_out_dir, rec)
            filename = os.path.join(rec_out_dir, rec + ".c3d")
            check_if_saved(filename)

        # Export real video
        if "video" in formats:
            export_real_video(subj_video_dir, rec)
            filename = os.path.join(subj_video_dir, rec + ".mp4")
            check_if_saved(filename)
            time.sleep(5)

        # Export 3D video
        if "video3D" in formats:
            export_3D_video(rec_out_dir, rec)
            filename = os.path.join(rec_out_dir, rec + ".mp4")
            check_if_saved(filename)

        # Export doc
        if "doc" in formats:
            export_doc(rec_out_dir, rec, subject)
            filename = os.path.join(rec_out_dir, rec + ".docx")
            check_if_saved(filename)
            activate_3DMA()

        logger(rec + " completed")

    logger(subject + " completed")
    print("-"*10 + "\n\n")

def activate_3DMA():
    '''
    Activate and maximize Clinical 3DMA Windows
    '''
    fw = pyautogui.getWindowsWithTitle("Clinical 3DMA")[0]
    fw.activate()
    time.sleep(.2)
    fw.maximize()

def main():
    # Check if 3DMA is running
    if not process_exists("MotionCaptor.exe"):
        logger("Clinical 3DMA is not running")
        return 0

    # Read config file
    try: 
        with open("config.json", "r") as config:
            config_file=json.load(config)
            output_data_dir = config_file["output_data_dir"]
            if not output_data_dir:
                logger("output_data_dir property not set")
                sys.exit()
            records = config_file["records"]
            formats = config_file["formats"]
    except OSError as e:
        logger("File config.json not found")
        sys.exit()

    # Check if output dir exists
    if not os.path.isdir(output_data_dir):
        logger(output_data_dir + " is not a valid directory")
        sys.exit()

    # Get screen resolution
    logger(pyautogui.size())

    # Start countdown
    countdown(4, text='Preparing to start')
    
    # Activate and maximize Clinical 3DMA Windows
    activate_3DMA()

    # Create video dir
    video_dir = os.path.join(output_data_dir, "video")
    if ("video" in formats) and (not os.path.isdir(video_dir)):
        os.makedirs(video_dir)

    for subject in records.keys():
        # Click on close capture
        pyautogui.moveTo(590, 90, duration=1)
        pyautogui.click(590, 90,)
        time.sleep(1)
        export_data(subject, records[subject], formats, output_data_dir, video_dir)
    
    pyautogui.alert(text='Your export is ready', title='Clinical 3DMA Exporter', button='OK')

    

if __name__ == "__main__":
    main()
    