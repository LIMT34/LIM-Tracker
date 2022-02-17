"""
CellPose 連携スクリプト
link to https://github.com/MouseLand/cellpose
"""
# coding: utf-8

import os
import sys
import json
import datetime
import numpy as np
import skimage.io
import glob
import random
import math
import zmq
import cv2
import struct
import tkinter as tk
from tkinter import *
from tkinter import messagebox
import tkinter.scrolledtext
import psutil
import threading
import time
import argparse
import socket
import configparser
import subprocess
import argparse, glob, pathlib
from natsort import natsorted
from tqdm import tqdm
from cellpose import utils, models, io
import logging
logger = logging.getLogger(__name__)

if os.name == 'nt':
    import win32api
    import win32event
    import winerror
    import pywintypes

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
    print("ROOT_DIR_2 ", ROOT_DIR)

sys.path.append(ROOT_DIR)  # To find local version of the library
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

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

def train_service():
    global channels
    global weightFolder
    if args.pretrained_model=='cyto' or args.pretrained_model=='nuclei' or args.pretrained_model=='cyto2':
        if args.mxnet and args.pretrained_model=='cyto2':
            logger.warning('cyto2 model not available in mxnet, using cyto model')
            args.pretrained_model = 'cyto'
        torch_str = ['torch', '']
        cpmodel_path = os.fspath(model_dir.joinpath('%s%s_0'%(args.pretrained_model, torch_str[args.mxnet])))
        if args.pretrained_model=='cyto':
            szmean = 30.
        else:
            szmean = 17.
    else:
        cpmodel_path = os.fspath(args.pretrained_model)
        szmean = 30.

    print("cpmodel_path: ",  cpmodel_path)
    test_dir = None if len(args.test_dir)==0 else args.test_dir
    output = io.load_train_test_data(args.dir, test_dir, imf, args.mask_filter, args.unet, args.look_one_level_down)
    images, labels, image_names, test_images, test_labels, image_names_test = output

    # training with all channels
    if args.all_channels:
        img = images[0]
        if img.ndim==3:
            nchan = min(img.shape)
        elif img.ndim==2:
            nchan = 1
        channels = None
    else:
        nchan = 2

    # model path
    if not os.path.exists(cpmodel_path):
        if not args.train:
            error_message = 'ERROR: model path missing or incorrect - cannot train size model'
            logger.critical(error_message)
            raise ValueError(error_message)
        cpmodel_path = False
        logger.info('>>>> training from scratch')
        if args.diameter==0:
            rescale = False
            logger.info('>>>> median diameter set to 0 => no rescaling during training')
        else:
            rescale = True
            szmean = args.diameter
    else:
        rescale = True
        args.diameter = szmean
        logger.info('>>>> pretrained model %s is being used'%cpmodel_path)
        args.residual_on = 1
        args.style_on = 1
        args.concatenation = 0
    if rescale and args.train:
        logger.info('>>>> during training rescaling images to fixed diameter of %0.1f pixels'%args.diameter)

    # initialize model
    if args.unet:
        model = core.UnetModel(device=device,
                                    pretrained_model=cpmodel_path,
                                    diam_mean=szmean,
                                    residual_on=args.residual_on,
                                    style_on=args.style_on,
                                    concatenation=args.concatenation,
                                    nclasses=args.nclasses,
                                    nchan=nchan)
    else:
        model = models.CellposeModel(device=device,
                                    torch=(not args.mxnet),
                                    pretrained_model=cpmodel_path,
                                    diam_mean=szmean,
                                    residual_on=args.residual_on,
                                    style_on=args.style_on,
                                    concatenation=args.concatenation,
                                    nchan=nchan)

    now = datetime.datetime.now()
    weightFolder = os.path.join(args.logs, "weight_{:%Y%m%d%H%M}".format(now))

    # train segmentation model
    if args.train:
        try:
            cpmodel_path = model.train(images, labels, train_files=image_names,
                                        test_data=test_images, test_labels=test_labels, test_files=image_names_test,
                                        learning_rate=args.learning_rate, channels=channels,
                                        save_path=weightFolder, #os.path.realpath(args.dir), #保存場所指定できる。
                                        rescale=rescale, n_epochs=args.n_epochs,
                                        batch_size=args.batch_size)
            model.pretrained_model = cpmodel_path
            logger.info('>>>> model trained and saved to %s'%cpmodel_path)

        except Exception as e:
            logger.error(e)
            messagebox.showerror('ERROR', e)
            print("ERROR train_service failed")

    # train size model
    if args.train_size:
        sz_model = models.SizeModel(cp_model=model, device=device)
        sz_model.train(images, labels, test_images, test_labels, channels=channels, batch_size=args.batch_size)
        if test_images is not None:
            predicted_diams, diams_style = sz_model.eval(test_images, channels=channels)
            if test_labels[0].ndim>2:
                tlabels = [lbl[0] for lbl in test_labels]
            else:
                tlabels = test_labels
            ccs = np.corrcoef(diams_style, np.array([utils.diameters(lbl)[0] for lbl in tlabels]))[0,1]
            cc = np.corrcoef(predicted_diams, np.array([utils.diameters(lbl)[0] for lbl in tlabels]))[0,1]
            logger.info('style test correlation: %0.4f; final test correlation: %0.4f'%(ccs,cc))
            np.save(os.path.join(args.test_dir, '%s_predicted_diams.npy'%os.path.split(cpmodel_path)[1]),
                    {'predicted_diams': predicted_diams, 'diams_style': diams_style})

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
def detect_service():
    global waitSocket
    global stop_flag
    global aftStart
    global titleLabel
    global channels
    global diameterText

    tic = time.time()
    if not (args.pretrained_model=='cyto' or args.pretrained_model=='nuclei' or args.pretrained_model=='cyto2'):
        cpmodel_path = args.pretrained_model
        if not os.path.exists(cpmodel_path):
            logger.warning('model path does not exist, using cyto model')
            args.pretrained_model = 'cyto'
    try:

        if args.pretrained_model=='cyto' or args.pretrained_model=='nuclei' or args.pretrained_model=='cyto2':
            if args.mxnet and args.pretrained_model=='cyto2':
                logger.warning('cyto2 model not available in mxnet, using cyto model')
                args.pretrained_model = 'cyto'
            model = models.Cellpose(gpu=gpu, device=device, model_type=args.pretrained_model, torch=(not args.mxnet))
        else:
            if args.all_channels:
                channels = None
            model = models.CellposeModel(gpu=gpu, device=device, pretrained_model=cpmodel_path, torch=(not args.mxnet))

        if args.diameter==0:
            if args.pretrained_model=='cyto' or args.pretrained_model=='nuclei' or args.pretrained_model=='cyto2':
                diameter = None
                logger.info('>>>> estimating diameter for each image')
            else:
                logger.info('>>>> using user-specified model, no auto-diameter estimation available')
                diameter = model.diam_mean
        else:
            diameter = args.diameter
            logger.info('>>>> using diameter %0.2f for all images'%diameter)

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
            logger.info('')
            logger.info("  READY!")
            logger.info('')
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

                diameter = int(diameterText.get())
                out = model.eval(image_cv, channels=channels, diameter=diameter,
                                    do_3D=args.do_3D, net_avg=(not args.fast_mode),
                                    augment=False,
                                    resample=args.resample,
                                    flow_threshold=args.flow_threshold,
                                    cellprob_threshold=args.cellprob_threshold,
                                    invert=args.invert,
                                    batch_size=args.batch_size,
                                    interp=(not args.no_interp),
                                    channel_axis=args.channel_axis,
                                    z_axis=args.z_axis)
                logger.info("  model.eval: size=" + str(col) + "/" + str(row) + ", mat_type=" + str(mat_type) + ", diameter=" + str(diameter)) #追加
                masks, flows = out[:2]
                if len(out) > 3:
                    diams = out[-1]
                else:
                    diams = diameter

                if args.exclude_on_edges:
                    masks = utils.remove_edge_masks(masks)

                masks = masks.astype(np.uint16)
                #cv2.imwrite("C:/CellTrackService/cellpose/test_mask.png", masks)

                height, width = masks.shape[:2]
                ndim = masks.ndim
                data = [ np.array( [height] ), np.array( [width] ), np.array( [ndim] ), masks.data ]
                responder.send_multipart(data)

        except Exception as e:
            logger.error(e)
            messagebox.showerror('ERROR', e)
            # ダミーデータを送信
            masks = np.zeros((row, col), np.uint16)
            data = [ np.array( [row] ), np.array( [col] ), np.array( [1] ), masks.data ]
            responder.send_multipart(data)
            print("ERROR detect_service failed")

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
def mainFunction():
    global stop_flag
    global weightFolder
    global model

    weightFolder = [" ----- "]
    StartTime = datetime.datetime.today()

    # Validate arguments
    if args.command == "train":
        logger.info('')
        logger.info("-----------------------------------------------------------");
        logger.info(" Training start: " + StartTime.strftime('%Y-%m-%d %H:%M:%S'))
        logger.info("-----------------------------------------------------------");
        logger.info('')
        train_service()

        EndTime = datetime.datetime.today()
        logger.info('')
        logger.info("-----------------------------------------------------------");
        logger.info(" Training end: " + EndTime.strftime('%Y-%m-%d %H:%M:%S'))
        logger.info("   --modeldir= " + weightFolder + "/models")
        logger.info("-----------------------------------------------------------");
        messagebox.showinfo('', 'Training process Completed')

    elif args.command == "detect":
        logger.info('')
        logger.info("-----------------------------------------------------------");
        logger.info(" Detection service start: " + StartTime.strftime('%Y-%m-%d %H:%M:%S'))
        logger.info("-----------------------------------------------------------");
        logger.info('')
        detect_service()

    else:
        logger.info("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))

def stopPolling():
    model.stopPolling()

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
def stop_close(event):
    global closeButton
    img = tk.PhotoImage(file=ROOT_DIR + "/icons/exit_program2.png")
    closeButton.configure(image=img)
    closeButton.photo = img
    stop()
    logger.info("ExitProgramButton Clicked!")

def stop_release(event):
    global closeButton
    img = tk.PhotoImage(file=ROOT_DIR + "/icons/exit_program1.png")
    closeButton.configure(image=img)
    closeButton.photo = img
    logger.info("ExitProgramButton Release!")

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
                logger.info('')
                logger.info("Stop the process and press the Exit button again.")
                logger.info('')

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
class CoreGUI(object):
    def __init__(self, parent):
        global titleLabel
        global processType
        global closeButton
        global diameterText

        self.parent = parent
        self.parent.title(" Cellpose: LIM Track Service 20211129")

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
            f04d = Frame(f04,width=15, height=10)
            f04d.grid(column = 1, row = 0, sticky = 'nsew')
            f04tb = Frame(f04,width=15, height=10)
            f04tb.grid(column = 1, row = 1, sticky = 'nsew')
            f04e = Frame(f04,width=15, height=10)
            f04e.grid(column = 1, row = 2, sticky = 'nsew')
            #--------------------------------------------------------

            f04f = Frame(f04,width=15, height=10)
            f04f.grid(column = 2, row = 0, sticky = 'nsew')
            f04g = Frame(f04,width=15, height=10)
            f04g.grid(column = 2, row = 1, sticky = 'nsew')
            f04h = Frame(f04,width=15, height=10)
            f04h.grid(column = 2, row = 2, sticky = 'nsew')

        if processType == "detect":
            #上段４（テンソルボードボタン）
            f04a = Frame(f04,width=15, height=20)
            f04a.grid(column = 0, row = 0, sticky = 'nsew')
            f04b = Frame(f04,width=15, height=20)
            f04b.grid(column = 0, row = 1, sticky = 'nsew')
            f04c = Frame(f04,width=15, height=20)
            f04c.grid(column = 0, row = 2, sticky = 'nsew')

            #--------------------------------------------------------
            f04d = Frame(f04, 				width=15, height=20)
            f04d.grid(						column = 0, row = 1, sticky = 'nsew')

            diameterLabel = Label(f04 ,		width=14, text="Diameter (pix): ", font=('Arial', 10), fg = 'gray30') #, bg = 'green'
            diameterLabel.grid(				column = 1, row = 1, sticky = 'nsew')

            #diameterText = tkinter.Entry(f04 ,	width=6, font=('Arial', 10), justify='center')
            #diameterText.grid(					column = 2, row = 1, sticky = 'nsew')

            current_value = tk.StringVar(value=30)
            diameterText = Spinbox(f04 ,	width=6, font=('Arial', 10), justify='center', textvariable=current_value,from_=1,to=10000,increment=1)
            diameterText.grid(					column = 2, row = 1, sticky = 'nsew')

            f04e = Frame(f04,				width=20, height=20)
            f04e.grid(						column = 3, row = 1, sticky = 'nsew')

            #--------------------------------------------------------


        #--------------------------------------------------------
        #中段（スクロールテキスト）
        SText = tkinter.scrolledtext.ScrolledText(self.parent,width=30, height=10)
        SText.grid(column = 0, row = 1, sticky = 'nsew')
        text_handler = TextHandler(SText)
        logging.basicConfig(filename='test.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger()
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
    global channels

    waitSocket = False;
    frame=1
    stop_flag=False
    thread=None
    aftStart = True

    if os.name == 'nt':
        UNIQUE_MUTEX_NAME = 'Global\\MyProgramIsAlreadyRunning'
        handle = win32event.CreateMutex(None, pywintypes.FALSE, UNIQUE_MUTEX_NAME)
        if not handle or win32api.GetLastError() == winerror.ERROR_ALREADY_EXISTS:
            messagebox.showerror('ERROR', 'Another process is already running.')
            logger.info('Another process is already running.')
            sys.exit(-1)

    parser = argparse.ArgumentParser(description='cellpose parameters')

    parser.add_argument('--logs', required=False, help='Logs')

    parser.add_argument("command", metavar="<command>", help="'train' or 'detect'")
    parser.add_argument('--check_mkl', action='store_true', help='check if mkl working')
    parser.add_argument('--mkldnn', action='store_true', help='for mxnet, force MXNET_SUBGRAPH_BACKEND = "MKLDNN"')
    parser.add_argument('--train', action='store_true', help='train network using images in dir')
    parser.add_argument('--dir', required=False, default=[], type=str, help='folder containing data to run or train on')
    parser.add_argument('--look_one_level_down', action='store_true', help='')
    parser.add_argument('--mxnet', action='store_true', help='use mxnet')
    parser.add_argument('--img_filter', required=False, default=[], type=str, help='end string for images to run on')
    parser.add_argument('--use_gpu', action='store_true', help='use gpu if mxnet with cuda installed')
    parser.add_argument('--fast_mode', action='store_true', help="make code run faster by turning off 4 network averaging")
    parser.add_argument('--resample', action='store_true', help="run dynamics on full image (slower for images with large diameters)")
    parser.add_argument('--no_interp', action='store_true', help='do not interpolate when running dynamics (was default)')
    parser.add_argument('--do_3D', action='store_true', help='process images as 3D stacks of images (nplanes x nchan x Ly x Lx')

    # settings for running cellpose
    parser.add_argument('--pretrained_model', required=False, default='cyto', type=str, help='model to use')
    parser.add_argument('--unet', required=False, default=0, type=int, help='run standard unet instead of cellpose flow output')
    parser.add_argument('--nclasses', required=False, default=3, type=int, help='if running unet, choose 2 or 3, otherwise not used')
    parser.add_argument('--chan', required=False, default=0, type=int, help='channel to segment; 0: GRAY, 1: RED, 2: GREEN, 3: BLUE')
    parser.add_argument('--chan2', required=False, default=0, type=int, help='nuclear channel (if cyto, optional); 0: NONE, 1: RED, 2: GREEN, 3: BLUE')
    parser.add_argument('--invert', required=False, action='store_true', help='invert grayscale channel')
    parser.add_argument('--all_channels', action='store_true', help='use all channels in image if using own model and images with special channels')
    parser.add_argument('--diameter', required=False, default=30., type=float, help='cell diameter, if 0 cellpose will estimate for each image')
    parser.add_argument('--flow_threshold', required=False, default=0.4, type=float, help='flow error threshold, 0 turns off this optional QC step')
    parser.add_argument('--cellprob_threshold', required=False, default=0.0, type=float, help='cell probability threshold, centered at 0.0')
    parser.add_argument('--save_png', action='store_true', help='save masks as png')
    parser.add_argument('--save_outlines', action='store_true', help='save outlines as text file for ImageJ')
    parser.add_argument('--save_tif', action='store_true', help='save masks as tif')
    parser.add_argument('--no_npy', action='store_true', help='suppress saving of npy')
    parser.add_argument('--channel_axis', required=False, default=None, type=int, help='axis of image which corresponds to image channels')
    parser.add_argument('--z_axis', required=False, default=None, type=int, help='axis of image which corresponds to Z dimension')
    parser.add_argument('--exclude_on_edges', action='store_true', help='discard masks which touch edges of image')

    # settings for training
    parser.add_argument('--train_size', action='store_true', help='train size network at end of training')
    parser.add_argument('--mask_filter', required=False, default='_masks', type=str, help='end string for masks to run on')
    parser.add_argument('--test_dir', required=False, default=[], type=str, help='folder containing test data (optional)')
    parser.add_argument('--learning_rate', required=False, default=0.2, type=float, help='learning rate')
    parser.add_argument('--n_epochs', required=False, default=500, type=int, help='number of epochs')
    parser.add_argument('--batch_size', required=False, default=8, type=int, help='batch size')
    parser.add_argument('--residual_on', required=False, default=1, type=int, help='use residual connections')
    parser.add_argument('--style_on', required=False, default=1, type=int, help='use style vector')
    parser.add_argument('--concatenation', required=False, default=0, type=int, help='concatenate downsampled layers with upsampled layers (off by default which means they are added)')

    args = parser.parse_args()
    processType = args.command

    if args.check_mkl:
        mkl_enabled = models.check_mkl((not args.mxnet))
    else:
        mkl_enabled = True

    if not args.train and (mkl_enabled and args.mkldnn):
        os.environ["MXNET_SUBGRAPH_BACKEND"]="MKLDNN"
    else:
        os.environ["MXNET_SUBGRAPH_BACKEND"]=""

    use_gpu = False
    channels = [args.chan, args.chan2]

    if len(args.img_filter)>0:
        imf = args.img_filter
    else:
        imf = None

    device, gpu = models.assign_device((not args.mxnet), args.use_gpu)
    model_dir = models.model_dir

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
