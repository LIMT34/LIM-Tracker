"""
Detectron2 連携スクリプト
link to https://github.com/facebookresearch/detectron2
"""
# coding: utf-8

import os
import sys
if os.name == 'nt':
    import win32api
    import win32event
    import winerror
    import pywintypes
import numpy as np
import json
import cv2
import math, random
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import time
from pathlib import Path
import argparse
import datetime
import configparser
import tkinter as tk
from tkinter import messagebox
import tkinter.simpledialog as simpledialog
from tkinter import *
import tkinter.scrolledtext
import psutil
import threading
import zmq
import socket
import glob
import struct
import logging
logger = logging.getLogger(__name__)

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
# Root directory of the project
if getattr(sys, 'frozen', False):
    ROOT_DIR = os.path.dirname(sys.executable) #EXE
    logger.info("ROOT_DIR_1 " + ROOT_DIR)
elif __file__:
    #print("ROOT_DIR_A ", os.getcwd())
    #print("ROOT_DIR_B ", os.path.dirname(__file__))
    ROOT_DIR = os.getcwd() #PY プロンプトから呼び出す場合
    TOOL_DIR = os.path.dirname(__file__)
    if TOOL_DIR != "":
        ROOT_DIR = TOOL_DIR
    logger.info("ROOT_DIR_2 " + ROOT_DIR)

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#log_dirをチェックし、h5ファイルがあればchromeを起動
def startTensorboard():
    global threadTensorboard
    global weightFolder
    global stop_flag
    stop_flag = False
    global end_flag
    end_flag = False

    if os.name == 'nt':
        command1 = "start tensorboard --logdir=" + weightFolder
        os.system(command1)
    else:
        command1 = "tensorboard --logdir=" + weightFolder
        subprocess.Popen(command1.split())

    print()
    print("-----------------------------------------------------------");
    print(" Tensorboard start")
    print("   --logdir=", weightFolder)
    print(" (If the graph is not displayed, press the reload button.)")
    print("-----------------------------------------------------------");
    print()

    flgH5File = False
    num = 0
    while num < 600:
        if num % 10 == 0:
            fileList = glob.glob(weightFolder + "/*.h5")
            t = len(fileList)
            if len(fileList) > 0:
                flgH5File = True
                time.sleep(5)
                break;
        if flgH5File:
            break;
        if stop_flag:
            break;
        num += 1
        time.sleep(1)
    if flgH5File:
        if os.name == 'nt':
            command2 = "start http://" + socket.gethostname() + ":6006/#scalars"
            os.system(command2)

    end_flag = True

def stopPolling():
    global stop_flag
    stop_flag = True
    return True

#def startTensorboardisEnd():
#    return end_flag

#---------------------------------------------------------------------------------
# python E:/celltrack_service.py
# --mode=train
# --dataset_name=phase_contrast_train
# --json_file=E:/datasets/phase_contrast/train/annotations.json
# --image_root=E:/datasets/phase_contrast/train
# --weights=default
# --output_dir=E:/datasets/weight_phase_contrast_mask_rcnn_X_101_32x8d_FPN_3x
#---------------------------------------------------------------------------------
def train_service():
    global weightFolder
    cfg = get_cfg()

    #データセット設定
    dataset_name = args.dataset_name #"phase_contrast_train"
    register_coco_instances(dataset_name, {}, args.json_file, args.image_root)
     #"E:/datasets/phase_contrast/train/annotations.json",
     #"E:/datasets/phase_contrast/train")

    #モデル設定
    config_file = args.config_file #"COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    #config_file = args.config_file #"COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.DATASETS.TRAIN = (dataset_name,) #位置かえるな
    cfg.DATASETS.TEST = () #2021/10/17追加

    #初期重み設定
    if args.weights=='default':
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file) #デフォルト重み設定
    else:
        cfg.MODEL.WEIGHTS = args.weights

    #エポック設定など
    cfg.SOLVER.MAX_ITER = args.max_iter # 1000000 #エポック回数は生成する重みファイルに埋め込まれている
    cfg.DATALOADER.NUM_WORKERS = args.num_workers # 2
    cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch #2
    cfg.SOLVER.BASE_LR = args.base_lr #0.00025
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.batch_size_per_image #512 #128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes #1

    #重みファイル出力先設定
    cfg.OUTPUT_DIR = args.output_dir #"E:/datasets/weight_phase_contrast_mask_rcnn_X_101_32x8d_FPN_3x"

    weightFolder = cfg.OUTPUT_DIR
    threadTensorboard = threading.Thread(target=startTensorboard)
    threadTensorboard.start()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    try:
        trainer.train()
    except Exception as e:
        logger.error(e)
        messagebox.showerror('ERROR', e)
        print("ERROR train_service failed")

#---------------------------------------------------------------------------------
# conda activate detectron2
# python E:/celltrack_service.py
# --mode=detect
# --config_file=COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml
# --weights=E:/datasets/weight_phase_contrast_mask_rcnn_X_101_32x8d_FPN_3x/model_0104999.pth
#---------------------------------------------------------------------------------
def detect_service():
    global waitSocket
    global stop_flag
    global aftStart
    global titleLabel

    cfg = get_cfg()
    #モデル設定
    config_file = args.config_file #"COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    #重み設定
    if args.weights=='default':
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file) #デフォルト重み設定
    else:
        cfg.MODEL.WEIGHTS = args.weights
    #クラス数
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes #1

    logger.info("config_file " + config_file)
    logger.info("cfg.MODEL.WEIGHTS " + cfg.MODEL.WEIGHTS)
    logger.info("cfg.MODEL.ROI_HEADS.NUM_CLASSES " + str(cfg.MODEL.ROI_HEADS.NUM_CLASSES))

    try:
        predictor = DefaultPredictor(cfg)
    except Exception as e:
        messagebox.showerror('ERROR', 'detect_service load_weights failed')
        print("ERROR detect_service load_weights failed")
        stop()

    ctx = zmq.Context()
    responder = ctx.socket(zmq.REP)
    responder.bind("tcp://*:11000")

    while not stop_flag:
        row = 0;
        col = 0;
        try:
            logger.info("")
            logger.info("  READY!")
            logger.info("")
            titleLabel["text"] = "Detection service"
            aftStart = False
            waitSocket = True;
            byte_rows, byte_cols, byte_mat_type, data=  responder.recv_multipart()
            waitSocket = False;
            if not stop_flag:
                row = struct.unpack('i', byte_rows)[0]
                col = struct.unpack('i', byte_cols)[0]
                mat_type = struct.unpack('i', byte_mat_type)
                if mat_type[0] != 3:
                    # Gray Scale
                    image_cv = np.frombuffer(data, dtype=np.uint8).reshape((row, col));
                else:
                    # BGR -> RGB Color
                    image_cv = np.frombuffer(data, dtype=np.uint8).reshape((row, col, 3));
                    image_cv = image_cv[:, :, [2, 1, 0]]

                outputs = predictor(image_cv)
                logger.info("  predictor: size=" + str(col) + "/" + str(row) + ", mat_type=" + str(mat_type)) #追加
                predictions = outputs["instances"].to("cpu")
                boxes = predictions.pred_boxes
                masks = np.asarray(predictions.pred_masks)

                roiCount = len(boxes)
                if not roiCount:
                    print("\n* No Roi * \n")
                masked_image = np.zeros((row, col), np.uint16) # マスク画像出力用バッファ作成
                for roiNo in range(roiCount):
                    x1 = (int)(boxes[roiNo].tensor[:, 0].item())
                    y1 = (int)(boxes[roiNo].tensor[:, 1].item())
                    x2 = (int)(boxes[roiNo].tensor[:, 2].item()+1.0)
                    y2 = (int)(boxes[roiNo].tensor[:, 3].item()+1.0)
                    print(" label" + str(roiNo+1) + " -> "  + str(x1) + ", " + str(y1) + ", " + str(x2) + ", " + str(y2))
                    mask = masks[roiNo][y1:y2+1,x1:x2+1]
                    masked_image[y1:y2+1,x1:x2+1] = np.where(mask == 1, (roiNo+1), masked_image[y1:y2+1,x1:x2+1])

                data = [ np.array( [row] ), np.array( [col] ), np.array( [1] ), masked_image.data ]
                responder.send_multipart(data)#送信

        except Exception as e:
            logger.error(e)
            messagebox.showerror('ERROR', e)
            # ダミーデータ送信
            masked_image = np.zeros((row, col), np.uint16) # マスク画像出力用バッファ作成
            data = [ np.array( [row] ), np.array( [col] ), np.array( [1] ), masked_image.data ]
            responder.send_multipart(data)#
            print("ERROR detect_service failed")

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
def mainFunction():
    global stop_flag
    StartTime = datetime.datetime.today()

    if args.mode == "train":
        logger.info('')
        logger.info("-----------------------------------------------------------");
        logger.info(" Training start: " + StartTime.strftime('%Y-%m-%d %H:%M:%S'))
        logger.info("-----------------------------------------------------------");
        logger.info('')
        train_service()
        #os.system(weightFolder[0])
        EndTime = datetime.datetime.today()
        logger.info('')
        logger.info("-----------------------------------------------------------");
        logger.info(" Training end: " + EndTime.strftime('%Y-%m-%d %H:%M:%S'))
        logger.info("-----------------------------------------------------------");
        #wait = input(" Press ENTER to quit")
        if os.name == 'nt':
            messagebox.showinfo('', 'Training process Completed')

    elif args.mode == "detect":
        logger.info('')
        logger.info("-----------------------------------------------------------");
        logger.info(" Detection service start: " + StartTime.strftime('%Y-%m-%d %H:%M:%S'))
        logger.info("-----------------------------------------------------------");
        logger.info('')
        detect_service()
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.mode))

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
def stop_close(event):
    global closeButton
    img = tk.PhotoImage(file=ROOT_DIR + "/icons/exit_program2.png")
    closeButton.configure(image=img)
    closeButton.photo = img
    stop()
    print("ExitProgramButton Clicked!")

def stop_release(event):
    global closeButton
    img = tk.PhotoImage(file=ROOT_DIR + "/icons/exit_program1.png")
    closeButton.configure(image=img)
    closeButton.photo = img
    print("ExitProgramButton Release!")

def stopProcess():
    sys.stdout = sys.__stdout__ #必要
    if os.name == 'nt':
        # EXE終了
        for proc in psutil.process_iter():
            if proc.name() == "tensorboard.exe" or proc.name() == "train_detect_service_gui.exe":
                for subproc in proc.children(recursive=True):
                    #print("Subproc1：" + subproc.name())
                    subproc.kill()
                proc.kill()
        #pyプロセス終了
        for proc in psutil.process_iter():
            try:
                ret0 = [s for s in proc.cmdline() if 'python' in s]
                ret1 = [s for s in proc.cmdline() if 'celltrack_service.py' in s]
                if len(ret0) == 1 and len(ret1) == 1:
                    for subproc in proc.children(recursive=True):
                        #print("Train Subproc: " + subproc.name())
                        subproc.kill()
                    proc.kill()
            except psutil.AccessDenied:
                print(psutil.AccessDenied)
    else:
        for p in psutil.process_iter(attrs=('name', 'pid', 'cmdline')):
            if "tensorboard" in p.info['name']:
                p = psutil.Process(p.info["pid"])
                p.kill()

        for p in psutil.process_iter(attrs=('name', 'pid', 'cmdline')):
            if 'python' == p.info['name'] and 'celltrack_service' in ' '.join(p.info['cmdline']):
                p = psutil.Process(p.info["pid"])
                p.kill()

    sys.exit(-1)

def stop():
    global waitSocket
    global stop_flag
    global thread
    global model
    global processType

    if processType == "train":
        ret = messagebox.askyesno('', 'Exit the program?')
        if ret == True:
            stopProcess()
    else:
        if waitSocket:
            ret = messagebox.askyesno('', 'Exit the program?')
            if ret == True:
                stop_flag=True
                conn_str="tcp://localhost:11000"
                args = sys.argv
                ctx = zmq.Context()
                sock = ctx.socket(zmq.REQ)
                sock.connect(conn_str)
                width = 0
                height = 0
                img = np.zeros((height, width, 3), np.uint8)
                data = [ np.array( [height] ), np.array( [width] ), np.array( [img.ndim] ), img.data ]
                sock.send_multipart(data)
                thread.join()
                thread=None
                stopProcess()
        else:
            if aftStart:
                stopProcess()
            else:
                messagebox.showerror('ERROR', 'Stop the process and press the Exit button again.')
                print()
                print("Stop the process and press the Exit button again.")
                print()

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
def chrome1_open(event):
    global tensorboardButton
    img = tk.PhotoImage(file=ROOT_DIR + "/icons/tensorboard2.png")
    tensorboardButton.configure(image=img)
    tensorboardButton.photo = img
    chrome1()
    #print("TensorBoard Clicked!")

def chrome1_release(event):
    global tensorboardButton
    img = tk.PhotoImage(file=ROOT_DIR + "/icons/tensorboard1.png")
    tensorboardButton.configure(image=img)
    tensorboardButton.photo = img
    #print("TensorBoard Release!")

def chrome1():
    thread2 = threading.Thread(target=chrome2)
    thread2.start()

def chrome2():
    global weightFolder
    if not weightFolder == " ----- ":

        flg = False
        if os.name == 'nt':
            for proc in psutil.process_iter():
                if proc.name() == "tensorboard.exe":
                    flg = True;
            if not flg:
                command1 = "start tensorboard --logdir=" + weightFolder
                os.system(command1)
                print()
                print("-----------------------------------------------------------");
                print(" Tensorboard start")
                print("   --logdir=", weightFolder)
                print(" (If the graph is not displayed, press the reload button.)")
                print("-----------------------------------------------------------");
                print()
                time.sleep(2)

            command2 = "start http://" + socket.gethostname() + ":6006/#scalars"
            os.system(command2)
        else:
            for proc in psutil.process_iter():
                if proc.name() == "tensorboard":
                    flg = True;
            if not flg:
                command1 = "tensorboard --logdir=" + weightFolder
                subprocess.Popen(command1.split())
                #os.system(command1)
                print()
                print("-----------------------------------------------------------");
                print(" Tensorboard start")
                print("   --logdir=", weightFolder)
                print(" (If the graph is not displayed, press the reload button.)")
                print("-----------------------------------------------------------");
                print()
                time.sleep(2)

            command2 = "http://localhost:6006/#scalars"
        stopPolling()
        #print(command2)

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
class CoreGUI(object):
    def __init__(self, parent):
        global titleLabel
        global processType
        global closeButton
        global tensorboardButton

        self.parent = parent
        self.parent.title(" Detectron2: LIM Track Service 20211129")

        if processType == "train":
            title = "Train the model"
            self.title = title
            self.parent.geometry("700x500")
        else:
            title = "Please wait..."
            self.title = title
            self.parent.geometry("500x600")

        #上段１
        canvas1 = Canvas(self.parent, width = 30, height = 30)
        canvas1.grid(column = 0, row = 0, sticky = 'nsew')

        canvas1 = Canvas(self.parent)
        canvas1.grid(column = 0, row = 0, sticky = 'nsew')

        #上段１（ラベル）
        f01 = Frame(canvas1,width=30)
        f01.grid(column = 0, row = 0, sticky = 'nsew')

        f01a = Frame(f01 ,width=15, height=15)
        f01a.grid(column = 0, row = 0, sticky = 'nsew')
        titleLabel = Label(f01 ,width=15, text=" " + title, font=('Arial', 10), fg = 'gray30') #, bg = 'green'
        titleLabel.grid(column = 0, row = 1, sticky = 'nsew')
        f01b = Frame(f01 ,width=15, height=15)
        f01b.grid(column = 0, row = 2, sticky = 'nsew')

        #上段２
        f02 = Frame(canvas1)
        f02.grid(column = 1, row = 0, sticky = 'nsew')

        #上段３
        f03 = Frame(canvas1)
        f03.grid(column = 2, row = 0, sticky = 'nsew')

        #上段４
        f04 = Frame(canvas1)
        f04.grid(column = 3, row = 0, sticky = 'nsew')

        if processType == "train":
            #上段４（テンソルボードボタン）
            f04a = Frame(f04,width=15, height=10)
            f04a.grid(column = 0, row = 0, sticky = 'nsew')
            f04b = Frame(f04,width=15, height=10)
            f04b.grid(column = 0, row = 1, sticky = 'nsew')
            f04c = Frame(f04,width=15, height=10)
            f04c.grid(column = 0, row = 2, sticky = 'nsew')

            #--------------------------------------------------------
            f04d = Frame(f04,height=10)
            f04d.grid(column = 1, row = 0, sticky = 'nsew')
            if os.name == 'nt':
                img = tk.PhotoImage(file=ROOT_DIR + "/icons/tensorboard1.png")
                tensorboardButton = Label(f04, image=img)
                tensorboardButton.grid(column = 1, row = 1, sticky = 'nsew')
                tensorboardButton.photo = img
                tensorboardButton.bind('<Button-1>', chrome1_open)
                tensorboardButton.bind('<ButtonRelease-1>', chrome1_release)
            f04e = Frame(f04,height=10)
            f04e.grid(column = 1, row = 2, sticky = 'nsew')
            #--------------------------------------------------------

            f04f = Frame(f04,width=15, height=10)
            f04f.grid(column = 2, row = 0, sticky = 'nsew')
            f04g = Frame(f04,width=15, height=10)
            f04g.grid(column = 2, row = 1, sticky = 'nsew')
            f04h = Frame(f04,width=15, height=10)
            f04h.grid(column = 2, row = 2, sticky = 'nsew')

        #--------------------------------------------------------
        #中段（スクロールテキスト）
        SText = tkinter.scrolledtext.ScrolledText(self.parent,width=30, height=10)
        SText.grid(column = 0, row = 1, sticky = 'nsew')
        text_handler = TextHandler(SText)
        logging.basicConfig(#filename='test.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s')
        logger.addHandler(text_handler)
        #--------------------------------------------------------

        #下段
        canvas2 = Canvas(self.parent)
        canvas2.grid(column = 0, row = 2, sticky = 'nsew')

        f05 = Frame(canvas2,width=30)
        f05.grid(column = 0, row = 0, sticky = 'nsew')
        f06 = Frame(canvas2,width=30)
        f06.grid(column = 1, row = 0, sticky = 'nsew')
        f07 = Frame(canvas2,width=30)
        f07.grid(column = 2, row = 0, sticky = 'nsew')
        f08 = Frame(canvas2,width=30)
        f08.grid(column = 3, row = 0, sticky = 'nsew')

        #右下Exitボタン
        f10 = Frame(f08,width=15, height=10)
        f10.grid(column = 0, row = 0, sticky = 'nsew')
        f11 = Frame(f08,width=15, height=10)
        f11.grid(column = 0, row = 1, sticky = 'nsew')
        f12 = Frame(f08,width=15, height=10)
        f12.grid(column = 0, row = 2, sticky = 'nsew')

        f13 = Frame(f08,height=10)
        f13.grid(column = 1, row = 0, sticky = 'nsew')

        img = tk.PhotoImage(file=ROOT_DIR + "/icons/exit_program1.png")
        closeButton = Label(f08, image=img)
        closeButton.grid(column = 1, row = 1, sticky = 'nsew')
        closeButton.photo = img
        closeButton.bind('<Button-1>', stop_close)
        closeButton.bind('<ButtonRelease-1>', stop_release)


        f15 = Frame(f08,height=10)
        f15.grid(column = 1, row = 2, sticky = 'nsew')

        f16 = Frame(f08,width=15, height=10)
        f16.grid(column = 2, row = 0, sticky = 'nsew')
        f17 = Frame(f08,width=15, height=10)
        f17.grid(column = 2, row = 1, sticky = 'nsew')
        f18 = Frame(f08,width=15, height=10)
        f18.grid(column = 2, row = 2, sticky = 'nsew')

        canvas1.grid_columnconfigure(0, weight = 1)
        canvas2.grid_columnconfigure(0, weight = 1)
        self.parent.grid_columnconfigure(0, weight = 1)
        self.parent.grid_rowconfigure(1, weight = 1)

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
class StdoutRedirector(object):
    def __init__(self,text_widget):
        self.text_space = text_widget
    def write(self,string):
        self.text_space.insert('end', string)
        self.text_space.see('end')
    def flush(self):
        pass

class TextHandler(logging.Handler):
    def __init__(self, text):
        logging.Handler.__init__(self)
        self.text = text

    def emit(self, record):
        msg = self.format(record)
        def append():
            self.text.configure(state='normal')
            self.text.insert(tk.END, msg + '\n')
            self.text.configure(state='disabled')
            self.text.yview(tk.END)
        self.text.after(0, append)

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
if __name__ == '__main__':
    global waitSocket
    global stop_flag
    global thread
    global processType
    global aftStart

    waitSocket = False;
    frame=1
    stop_flag=False
    thread=None
    aftStart = True

    logger = setup_logger()

    if os.name == 'nt':
        UNIQUE_MUTEX_NAME = 'Global\\MyProgramIsAlreadyRunning'
        handle = win32event.CreateMutex(None, pywintypes.FALSE, UNIQUE_MUTEX_NAME)
        if not handle or win32api.GetLastError() == winerror.ERROR_ALREADY_EXISTS:
            messagebox.showerror('ERROR', 'Another process is already running.')
            print('Another process is already running.', file=sys.stderr)
            sys.exit(-1)

    parser = argparse.ArgumentParser(description='Detectron2 Training Script')
    parser.add_argument('--mode', default='train', type=str, help='train or detect')
    parser.add_argument('--config_file', default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", type=str, help='')

    parser.add_argument('--dataset_name', default=None, type=str, help='')
    parser.add_argument('--json_file', default=None, type=str, help='')
    parser.add_argument('--image_root', default=None, type=str, help='')

    parser.add_argument('--weights', default="default", type=str, help='')
    parser.add_argument('--output_dir', default=1, type=str, help='')

    #学習
    parser.add_argument('--num_workers', default=2, type=int, help='')
    parser.add_argument('--ims_per_batch', default=2, type=int, help='')
    parser.add_argument('--base_lr', default=0.00025, type=float, help='')
    parser.add_argument('--max_iter', default=100000, type=int, help='')
    parser.add_argument('--batch_size_per_image', default=512, type=int, help='')
    parser.add_argument('--num_classes', default=1, type=int, help='')

    #認識
    parser.add_argument('--score_thresh_test', default=0.7, type=float, help='') # set the testing threshold for this model

    args = parser.parse_args()

    processType = args.mode

    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", stop)
    gui = CoreGUI(root)

    img = tk.PhotoImage(file=ROOT_DIR + "/icons/Configure.png")
    root.tk.call('wm', 'iconphoto', root._w, img)

    if not thread:
        thread = threading.Thread(target=mainFunction)
        stop_flag=False
        thread.start()

    root.mainloop()
