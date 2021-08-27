#! /usr/bin/env python3
'''Clinical 3DMA GUI automation to facilitate and speed
data export process.

IMPORTANT:
This script is only compatible with Clinical 3DMA OT 2020.1
running on a 1920x1080 screen
'''
import time
import sys
import os
import shutil
import json
import datetime
import pyautogui
import psutil
import gui_coordinates as cord

# The directory where Clinical 3DMA stores the Word reports it generates.
STT_REPORTS = "C:/Users/WeCareLab/Documents/STT/3DMA/Reports"
# Clinical 3DMA windows title
DMA_WINDOWS_TITLE = "Clinical 3DMA"
# Clinical 3DMA database windows title
DMA_DATABASE_TITLE = "Base de datos"
# Clinical 3DMA process name
DMA_PROCESS_NAME = "MotionCaptor.exe"


def process_exists(process_name):
    '''Check if there is any running process that contains
    the given name process_name'''
    return process_name in (p.name() for p in psutil.process_iter())


def countdown(seconds, text=''):
    '''Countdown timer'''
    while seconds:
        timer = '%s %d' % (text, seconds)
        print(timer, end="\r")
        time.sleep(1)
        seconds -= 1
    logger("Starting")


def load_user(user_id):
    '''GUI automation to find and select an user with given
    user_id in the Clinical 3DM database'''

    # Click on "Select user"
    pyautogui.click(cord.SELECT_USER)
    time.sleep(2)

    # Maximice database windows
    activate_database()

    # Click on search
    pyautogui.click(cord.BD_SEARCH_USER)

    # Type username
    pyautogui.typewrite(user_id)

    # Select user
    pyautogui.moveTo(cord.BD_SELECT_USER, duration=1)
    pyautogui.click(cord.BD_SELECT_USER)

    # Confirm
    pyautogui.moveTo(cord.BD_CONFIRM, duration=1)
    pyautogui.click(cord.BD_CONFIRM)

    time.sleep(2)


def open_record(record_id):
    '''GUI automation to find and select a 3D motion capture
    with given record_id in the Clinical 3DM database'''
    # Click on "Open Capture"
    pyautogui.click(cord.OPEN_CAPTURE)
    time.sleep(2)

    # Maximize Database windows
    activate_database()

    # Click on search
    pyautogui.moveTo(cord.BD_SEARCH_CAPTURE, duration=1)
    pyautogui.click(cord.BD_SEARCH_CAPTURE)

    # Type record name
    pyautogui.typewrite(record_id)

    # Select record
    pyautogui.moveTo(cord.BD_SELECT_CAPTURE, duration=1)
    pyautogui.click(cord.BD_SELECT_CAPTURE)

    # Confirm
    pyautogui.moveTo(cord.BD_CONFIRM, duration=1)
    pyautogui.click(cord.BD_CONFIRM)

    time.sleep(10)


def handle_file_explorer(path, filename=None):
    '''GUI automation for saving a file with given filename at specific
    path using Windows 10 File Explorer'''
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
    ''''GUI automation to export full motion capture file (3MC)'''
    # Click on export data
    pyautogui.moveTo(cord.EXPORT_DATA, duration=1)
    pyautogui.click(cord.EXPORT_DATA)

    # Click on export 3MC
    pyautogui.moveTo(cord.EXPORT_3MC, duration=1)
    pyautogui.click(cord.EXPORT_3MC)
    time.sleep(2)

    # File explorer dialog
    handle_file_explorer(path, filename)

    time.sleep(.5)


def export_3dt(path, filename):
    '''GUI automation to export 3DT file'''
    # Click on export data
    pyautogui.moveTo(cord.EXPORT_DATA, duration=1)
    pyautogui.click(cord.EXPORT_DATA)

    # Click on export 3DT
    pyautogui.moveTo(cord.EXPORT_3DT, duration=1)
    pyautogui.click(cord.EXPORT_3DT)
    time.sleep(2)

    # File explorer dialog
    handle_file_explorer(path, filename)

    time.sleep(.5)


def export_trc(path, filename):
    '''GUI automation to export marker trajectories in TRC format'''
    # Click on export data
    pyautogui.moveTo(cord.EXPORT_DATA, duration=1)
    pyautogui.click(cord.EXPORT_DATA)

    # Click on export TRC
    pyautogui.moveTo(cord.EXPORT_TRC, duration=1)
    pyautogui.click(cord.EXPORT_TRC)
    time.sleep(2)

    # Open on file explorer
    pyautogui.moveTo(cord.EXPORT_TRC_EXPLORER, duration=1)
    pyautogui.click(cord.EXPORT_TRC_EXPLORER)
    time.sleep(.3)

    # File explorer dialog
    handle_file_explorer(path, filename)

    # Click on accept
    pyautogui.moveTo(cord.EXPORT_TRC_OK, duration=1)
    pyautogui.click(cord.EXPORT_TRC_OK)

    time.sleep(.5)


def export_c3d(path, filename):
    '''GUI automation to export motion capture in C3D format'''
    # Click on export data
    pyautogui.moveTo(cord.EXPORT_DATA, duration=1)
    pyautogui.click(cord.EXPORT_DATA)

    # Click on export C3D
    pyautogui.moveTo(cord.EXPORT_C3D, duration=1)
    pyautogui.click(cord.EXPORT_C3D,)
    time.sleep(1)

    # Click on accept
    pyautogui.moveTo(cord.EXPORT_C3D_OK, duration=1)
    pyautogui.click(cord.EXPORT_C3D_OK)
    time.sleep(.5)

    # File explorer dialog
    handle_file_explorer(path, filename)

    time.sleep(.5)


def export_raw(path, filename):
    '''GUI automation to export marker trajectories in TXT format'''
    # Click on export data
    pyautogui.moveTo(cord.EXPORT_DATA, duration=1)
    pyautogui.click(cord.EXPORT_DATA)

    # Click on export RAW
    pyautogui.moveTo(cord.EXPORT_RAW, duration=1)
    pyautogui.click(cord.EXPORT_RAW)
    time.sleep(2)

    # File explorer dialog
    handle_file_explorer(path, filename + '.raw')

    time.sleep(.5)


def export_events(path, filename):
    '''GUI automation to export a TXT file with
    relevant events time marks'''
    # Click on export data
    pyautogui.moveTo(cord.EXPORT_DATA, duration=1)
    pyautogui.click(cord.EXPORT_DATA)

    # Click on export events
    pyautogui.moveTo(cord.EXPORT_EVENTS, duration=1)
    pyautogui.click(cord.EXPORT_EVENTS)
    time.sleep(2)

    # File explorer dialog
    handle_file_explorer(path, filename + '.events')

    time.sleep(.5)


def export_doc(path, filename, subject):
    '''GUI automation to generate and export a Microsoft Word report'''
    doc_path = os.path.join(STT_REPORTS, subject + '_' + filename + '_1.docx')
    if not os.path.isfile(doc_path):
        # Click on generate report
        pyautogui.moveTo(cord.CREATE_REPORT, duration=1)
        pyautogui.click(cord.CREATE_REPORT)

        # Select walking
        pyautogui.click(cord.CREATE_REPORT_SELECT)
        time.sleep(.2)
        pyautogui.click(cord.CREATE_REPORT_WALKING)
        time.sleep(.2)
        pyautogui.moveTo(cord.CREATE_REPORT_OK, duration=1)
        pyautogui.click(cord.CREATE_REPORT_OK)

        # Wait for doc
        time.sleep(15)
        while True:

            # Get active windows
            actwd1 = pyautogui.getActiveWindow()
            if (os.path.isfile(doc_path)) and ("Office" in actwd1.title or "Word" in actwd1.title):
                time.sleep(10)
                # Close Office license windows
                if "Office" in actwd1.title:
                    pyautogui.moveTo(cord.MICROSOFT_OFFICE_LICENSE, duration=1)
                    pyautogui.click(cord.MICROSOFT_OFFICE_LICENSE)
                    time.sleep(1)

                pyautogui.moveTo(cord.MICROSOFT_OFFICE_CLOSE, duration=1)
                pyautogui.click(cord.MICROSOFT_OFFICE_CLOSE)
                break

            time.sleep(10)

    time.sleep(5)

    # Copy report to record directory
    shutil.copy2(doc_path, (path + '/' + filename + '.docx'))


def logger(text):
    '''Logger for the screen'''
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("[" + current_time + "] ", text)


def export_real_video(path, filename):
    '''GUI automation to export a synchronised video'''
    # Click on export data
    pyautogui.moveTo(cord.EXPORT_DATA, duration=1)
    pyautogui.click(cord.EXPORT_DATA)
    time.sleep(.2)

    # Click on export
    pyautogui.moveTo(cord.EXPORT_VIDEO_FEED, duration=1)
    pyautogui.click(cord.EXPORT_VIDEO_FEED)

    time.sleep(.4)

    # File explorer dialog
    handle_file_explorer(path, filename)

    # wait for video generation
    time.sleep(15)
    video_path = os.path.join(path, filename + '.mp4')
    while True:
        # Try to minimize Clinical 3DMA
        pyautogui.moveTo(cord.MINIMIZE_3DMA, duration=1)
        pyautogui.click(cord.MINIMIZE_3DMA)

        # Get active windows
        actwd1 = pyautogui.getActiveWindow()

        if (os.path.isfile(video_path) and (not actwd1.title.startswith(DMA_WINDOWS_TITLE))):
            activate_3dma()
            break

        time.sleep(15)

    time.sleep(.5)


def export_3d_video(path, filename):
    '''GUI automation to export a video file with 3D view'''
    # Click on export data
    pyautogui.moveTo(cord.EXPORT_DATA, duration=1)
    pyautogui.click(cord.EXPORT_DATA)
    time.sleep(.2)

    # Click on export 3D video
    pyautogui.moveTo(cord.EXPORT_VIDEO_3D, duration=1)
    pyautogui.click(cord.EXPORT_VIDEO_3D)

    time.sleep(.4)

    # File explorer dialog
    handle_file_explorer(path, filename)

    # wait for video generation
    time.sleep(40)
    while True:
        # Check if active windows is Clinical 3DMA
        actwd1 = pyautogui.getActiveWindow()

        if actwd1.title == 'Exportando...':
            time.sleep(40)
        else:
            break

    time.sleep(.5)


def export_strides(path):
    '''GUI automation to export kinematics data from stride dialog'''
    # Click on select stride
    pyautogui.moveTo(cord.STRIDE_SELECTION, duration=1)
    pyautogui.click(cord.STRIDE_SELECTION)

    time.sleep(3)

    # Click on export strides
    pyautogui.moveTo(cord.EXPORT_STRIDES, duration=1)
    pyautogui.click(cord.EXPORT_STRIDES)
    time.sleep(2)

    # File explorer dialog
    handle_file_explorer(path)

    # click on OK
    pyautogui.moveTo(cord.EXPORT_STRIDES_OK, duration=1)
    pyautogui.click(cord.EXPORT_STRIDES_OK)

    time.sleep(.5)


def export_biomechanics(path):
    '''GUI automation to export CSV files with biomechanical data'''
    # Click on export biomechanics
    pyautogui.moveTo(cord.EXPORT_BIOMECHANICS, duration=1)
    pyautogui.click(cord.EXPORT_BIOMECHANICS)

    time.sleep(2)

    # File explorer dialog
    handle_file_explorer(path)

    time.sleep(3)


def check_if_saved(filename):
    '''Check if a File with a given filename exists'''
    if os.path.isfile(filename):
        logger(filename + ' saved')
    else:
        logger(filename + " not saved")
        sys.exit()


def export_data(subject, recs, formats, output_dir, video_dir):
    '''Exports a set of motion captures ("recs") for a user with id "subject".
    For each record, save the specified "formats" in the output directory
    "output_dir". If there is a synchronised video, it is exported to
    "video_dir"'''
    logger(subject)
    subj_out_dir = os.path.join(output_dir, subject)

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
        rec_out_dir = rec_out_dir = os.path.join(subj_out_dir, rec.split(
            subject + "_")[1]) if rec.startswith(subject) else os.path.join(subj_out_dir, rec)

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
            filename = os.path.join(
                strides_out_dir, "Abducción aducción de cadera_Derecha.csv")
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
            export_3d_video(rec_out_dir, rec)
            filename = os.path.join(rec_out_dir, rec + ".mp4")
            check_if_saved(filename)

        # Export doc
        if "doc" in formats:
            export_doc(rec_out_dir, rec, subject)
            filename = os.path.join(rec_out_dir, rec + ".docx")
            check_if_saved(filename)
            activate_3dma()

        logger(rec + " completed")

    logger(subject + " completed")
    print("-"*10 + "\n\n")


def activate_3dma():
    '''
    Activate and maximize main windows of Clinical 3DMA
    '''
    windows_3dma = pyautogui.getWindowsWithTitle(DMA_WINDOWS_TITLE)[0]
    windows_3dma.activate()
    time.sleep(.2)
    windows_3dma.maximize()


def activate_database():
    '''
    Activate and maximize windows of Clinical 3DMA Database
    '''
    windows_3dma = pyautogui.getWindowsWithTitle(DMA_DATABASE_TITLE)

    if not windows_3dma:
        time.sleep(2)

    windows_3dma[0].activate()
    time.sleep(.5)
    windows_3dma[0].maximize()


def main():
    '''Script entrypoint.
    Check if Clinical 3DMA is running on a screen with the supported resolution.
    Then, load the settings file containing a list of captures and
    formats to be exported.
    '''
    # Check if 3DMA is running
    if not process_exists(DMA_PROCESS_NAME):
        logger("Clinical 3DMA is not running")
        return

    # Read config file
    try:
        with open("config.json", "r") as config:
            config_file = json.load(config)
            output_data_dir = config_file["output_data_dir"]
            if not output_data_dir:
                logger("output_data_dir property not set")
                sys.exit()
            records = config_file["records"]
            formats = config_file["formats"]
    except OSError:
        logger("File config.json not found")
        sys.exit()

    # Check if output dir exists
    if not os.path.isdir(output_data_dir):
        logger(output_data_dir + " is not a valid directory")
        sys.exit()

    # Check  screen resolution
    screen_size = pyautogui.size()
    if not (screen_size.width == 1920 and screen_size.height == 1080):
        logger(str(screen_size.width) + "x" + str(screen_size.height) +
               " is not a valid screen resolution")
        sys.exit()

    # Start countdown
    countdown(4, text='Preparing to start')

    # Activate and maximize Clinical 3DMA Windows
    activate_3dma()

    # Create video dir
    video_dir = os.path.join(output_data_dir, "video")
    if ("video" in formats) and (not os.path.isdir(video_dir)):
        os.makedirs(video_dir)

    for subject in records.keys():
        # Click on close capture
        pyautogui.moveTo(cord.CLOSE_CAPTURE, duration=1)
        pyautogui.click(cord.CLOSE_CAPTURE)
        time.sleep(1)
        export_data(subject, records[subject],
                    formats, output_data_dir, video_dir)

    pyautogui.alert(text='Your export is ready',
                    title='Clinical 3DMA Exporter', button='OK')


if __name__ == "__main__":
    main()
