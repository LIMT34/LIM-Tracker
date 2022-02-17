"""
Startdist 連携スクリプト
link to https://github.com/stardist/stardist
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

#Stardist
from stardist.models import StarDist2D
#from __future__ import print_function, unicode_literals, absolute_import, division
from csbdeep.utils import Path, normalize
from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.matching import matching, matching_dataset
from stardist.models import Config2D, StarDist2D, StarDistData2D
from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label

np.random.seed(42)
lbl_cmap = random_label_cmap()

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
    print("ROOT_DIR_1 " + ROOT_DIR)
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


#データセットの画像ファイル名取得等
def get_image_files(folder, mask_filter, imf=None, look_one_level_down=False):
    """ find all images in a folder and if look_one_level_down all subfolders """
    mask_filters = ['_cp_masks', '_cp_output', '_flows', mask_filter]
    image_names = []
    if imf is None:
        imf = ''

    folders = []
    if look_one_level_down:
        folders = natsorted(glob.glob(os.path.join(folder, "*/")))
    folders.append(folder)

    for folder in folders:
        image_names.extend(glob.glob(folder + '/*%s.png'%imf))
        image_names.extend(glob.glob(folder + '/*%s.jpg'%imf))
        image_names.extend(glob.glob(folder + '/*%s.jpeg'%imf))
        image_names.extend(glob.glob(folder + '/*%s.tif'%imf))
        image_names.extend(glob.glob(folder + '/*%s.tiff'%imf))
    image_names = natsorted(image_names)
    imn = []
    for im in image_names:
        imfile = os.path.splitext(im)[0]
        igood = all([(len(imfile) > len(mask_filter) and imfile[-len(mask_filter):] != mask_filter) or len(imfile) < len(mask_filter)
                        for mask_filter in mask_filters])
        if len(imf)>0:
            igood &= imfile[-len(imf):]==imf
        if igood:
            imn.append(im)
    image_names = imn

    if len(image_names)==0:
        raise ValueError('ERROR: no images in --dir folder')

    return image_names

#データセットの画像ファイル名取得等
def get_label_files(image_names, mask_filter, imf=None):
    nimg = len(image_names)
    label_names0 = [os.path.splitext(image_names[n])[0] for n in range(nimg)]

    if imf is not None and len(imf) > 0:
        label_names = [label_names0[n][:-len(imf)] for n in range(nimg)]
    else:
        label_names = label_names0

    # check for masks
    if os.path.exists(label_names[0] + mask_filter + '.tif'):
        label_names = [label_names[n] + mask_filter + '.tif' for n in range(nimg)]
    elif os.path.exists(label_names[0] + mask_filter + '.png'):
        label_names = [label_names[n] + mask_filter + '.png' for n in range(nimg)]
    else:
        raise ValueError('labels not provided with correct --mask_filter')
    if not all([os.path.exists(label) for label in label_names]):
        raise ValueError('labels not provided for all images in train and/or test set')

    return label_names

#データセットの画像ファイル読込み
def load_train_test_data(train_dir, image_filter=None, mask_filter='_masks', look_one_level_down=True):
    image_names = get_image_files(train_dir, mask_filter, image_filter, look_one_level_down)
    nimg = len(image_names)
    images = [cv2.imread(image_names[n], cv2.IMREAD_ANYDEPTH) for n in range(nimg)]
    #[print(image_names[n]) for n in range(nimg)]
    label_names = get_label_files(image_names, mask_filter, imf=image_filter)
    labels = [cv2.imread(label_names[n], cv2.IMREAD_ANYDEPTH) for n in range(nimg)]
    #[print(label_names[n]) for n in range(nimg)]
    return images, labels, image_names

def random_fliprot(img, mask):
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim)))
    mask = mask.transpose(perm)
    for ax in axes:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask

def random_intensity_change(img):
    img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
    return img

def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    # add some gaussian noise
    sig = 0.02*np.random.uniform(0,1)
    x = x + sig*np.random.normal(0,1,x.shape)
    return x, y

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
    output = load_train_test_data(args.dataset, '_img', '_masks', True)
    X, Y, image_names = output
    #print(type(X[0]))#numpy.ndarray 原画像
    #print(X[0].dtype)#uint8
    #print(type(Y[0]))#numpy.ndarray マスク画像
    #print(Y[0].dtype)#uint16

    n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]

    #画像を正規化し、小さなラベルの穴を埋める
    axis_norm = (0,1)   # normalize channels independently
    # axis_norm = (0,1,2) # normalize channels jointly
    if n_channel > 1:
        print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
        sys.stdout.flush()

    X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X)]
    Y = [fill_label_holes(y) for y in tqdm(Y)]

    #訓練データセットと検証データセットに分割
    assert len(X) > 1, "not enough training data"
    rng = np.random.RandomState(42)
    ind = rng.permutation(len(X))
    n_val = max(1, int(round(0.15 * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    X_val, Y_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val]
    X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]
    print('number of images: %3d' % len(X))
    print('- training:       %3d' % len(X_trn))
    print('- validation:     %3d' % len(X_val))

    i = min(9, len(X)-1)
    img, lbl = X[i], Y[i]
    assert img.ndim in (2,3)
    img = img if (img.ndim==2 or img.shape[-1]==3) else img[...,0]
    #plot_img_label(img,lbl)
    #None;

    #コンフィグレーション
    #StarDist2Dのモデルは、Config2Dオブジェクトによって指定されます。
    #print(Config2D.__doc__)

    # 32 is a good default choice (see 1_data.ipynb)
    n_rays = 32

    # Use OpenCL-based computations for data generator during training (requires 'gputools')
    use_gpu = False and gputools_available()

    # Predict on subsampled grid for increased efficiency and larger field of view
    grid = (2,2)

    conf = Config2D (
        n_rays       = n_rays,
        grid         = grid,
        use_gpu      = use_gpu,
        n_channel_in = n_channel,
    )
    #print(conf)
    vars(conf)
    try:
        if use_gpu:
            from csbdeep.utils.tf import limit_gpu_memory
            # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
            limit_gpu_memory(0.8)
            # alternatively, try this:
            # limit_gpu_memory(None, allow_growth=True)

        folderpath = args.model_save_folder
        tmp = os.path.split(folderpath)
        _basedir = tmp[0]
        _name = tmp[1]
        model = StarDist2D(conf, name=_name, basedir=_basedir)#---------------------------

        #Using default values: prob_thresh=0.5, nms_thresh=0.4.
        #Check if the neural network has a large enough field of view to see up to the boundary of most objects.
        #デフォルト値を使用: prob_thresh=0.5, nms_thresh=0.4.
        median_size = calculate_extents(list(Y), np.median)
        fov = np.array(model._axes_tile_overlap('YX'))
        #print(f"median object size:      {median_size}")
        #print(f"network field of view :  {fov}")
        if any(median_size > fov):
            print("WARNING: median object size larger than field of view of the neural network.")

        #Data Augmentation
        # plot some augmented examples
        img, lbl = X[0],Y[0]
        for _ in range(3):
            img_aug, lbl_aug = augmenter(img,lbl)
            #plot_img_label(img_aug, lbl_aug, img_title="image augmented", lbl_title="label augmented")

        #トレーニング
        weightFolder = args.model_save_folder
        threadTensorboard = threading.Thread(target=startTensorboard)
        threadTensorboard.start()

        #$ tensorboard --logdir=. ブラウザで http://localhost:6006/ に接続します。

        history = model.train(X_trn,
                    Y_trn,
                    validation_data=(X_val,Y_val),
                    augmenter=augmenter,
                    epochs=args.epochs,                     #---------------------------
                    steps_per_epoch=args.stepperepoch)      #---------------------------
        model.export_TF() #-----------------------ImageJ用重み出力
        
    except Exception as e:
        print(e)
        messagebox.showerror('ERROR', e)
        print("ERROR train_service failed")

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
def detect_service():
    global waitSocket
    global stop_flag
    global aftStart
    global titleLabel
    global channels

    tic = time.time()

    #Stardist
    #You can access these pretrained models from stardist.models.StarDist2D
    # prints a list of available pretrained models
    StarDist2D.from_pretrained()

    try:
        if args.pretrained_model == '2D_versatile_fluo' or args.pretrained_model == '2D_versatile_he' or args.pretrained_model == '2D_paper_dsb2018' or args.pretrained_model == '2D_demo':
            # creates a pretrained model
            model = StarDist2D.from_pretrained(args.pretrained_model)
        elif args.pretrained_model == 'default':
            model = StarDist2D.from_pretrained('2D_versatile_fluo')
            print("detect_service default")
        else:
            folderpath = args.pretrained_model
            tmp = os.path.split(folderpath)
            _basedir = tmp[0]
            _name = tmp[1]
            print("model = StarDist2D(" + _basedir + ", " + _name + ")")
            model = StarDist2D(None, name=_name, basedir=_basedir)
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
            print('')
            print("  READY!")
            print('')
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
                    #print(" -> Gray Scale")
                else:
                    # BGR -> RGB Color
                    image_cv = np.frombuffer(data, dtype=np.uint8).reshape((row, col, 3));
                    image_cv = image_cv[:, :, [1]]
                    #print(" -> BGR -> RGB Color")
                    cv2.imwrite("C:/CellTrackService/stardist/test_in.png", image_cv)
                    #print("in -> ", image_cv.dtype)

                masks, _ = model.predict_instances(normalize(image_cv))
                print("  model.predict_instances: size=" + str(col) + "/" + str(row) + ", mat_type=" + str(mat_type)) #追加
                masked_image = masks.astype(np.uint16)#int32 -> uint16

                #cv2.imwrite("C:/CellTrackService/stardist/test_mask.png", masked_image)

                data = [ np.array( [row] ), np.array( [col] ), np.array( [1] ), masked_image.data ]

                responder.send_multipart(data)#送信

        except Exception as e:
            print(e)
            messagebox.showerror('ERROR', e)
            print("ERROR detect_service failed")
            # ダミーデータ送信
            masks = np.zeros((row, col), np.uint16)
            data = [ np.array( [row] ), np.array( [col] ), np.array( [1] ), masks.data ]
            responder.send_multipart(data)

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
        print('')
        print("-----------------------------------------------------------");
        print(" Training start: " + StartTime.strftime('%Y-%m-%d %H:%M:%S'))
        print("-----------------------------------------------------------");
        print('')
        train_service()

        EndTime = datetime.datetime.today()
        print('')
        print("-----------------------------------------------------------");
        print(" Training end: " + EndTime.strftime('%Y-%m-%d %H:%M:%S'))
        print("   --modeldir= " + args.model_save_folder)
        print("-----------------------------------------------------------");
        messagebox.showinfo('', 'Training process Completed')

    elif args.command == "detect":
        print('')
        print("-----------------------------------------------------------");
        print(" Detection service start: " + StartTime.strftime('%Y-%m-%d %H:%M:%S'))
        print("-----------------------------------------------------------");
        print('')
        detect_service()

    else:
        print("'{}' is not recognized. "
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
    print("ExitProgramButton Clicked!")

def stop_release(event):
    global closeButton
    img = tk.PhotoImage(file=ROOT_DIR + "/icons/exit_program1.png")
    closeButton.configure(image=img)
    closeButton.photo = img
    print("ExitProgramButton Release!")

def stopProcess():
    sys.stdout = sys.__stdout__ #必要！！！！
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

        self.parent = parent
        self.parent.title(" StarDist: LIM Track Service 20211129")

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
        sys.stdout = StdoutRedirector(SText)#標準出力をスクロールテキストにリダイレクト

        #text_handler = TextHandler(SText)
        #logging.basicConfig(filename='test.log',
        #    level=logging.INFO,
        #    format='%(asctime)s - %(levelname)s - %(message)s')
        #logger = logging.getLogger()
        #logger.addHandler(text_handler)
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

        #右下のExitボタン
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
            print('Another process is already running.')
            sys.exit(-1)

    parser = argparse.ArgumentParser(description='stardist parameters')
    parser.add_argument("command", metavar="<command>", help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False, default=[], type=str, help='folder containing data to run or train on')
    parser.add_argument('--pretrained_model', required=False, default='cyto', type=str, help='model to use')
    parser.add_argument('--model_save_folder', required=False, default='cyto', type=str, help='save model')
    parser.add_argument('--epochs', required=False, default=500, type=int, help='number of epochs')
    parser.add_argument('--stepperepoch', required=False, default=8, type=int, help='step per epoch')
    parser.add_argument('--learning_rate', required=False, default=0.2, type=float, help='learning rate')

    args = parser.parse_args()
    processType = args.command


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
