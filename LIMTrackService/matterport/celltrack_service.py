"""
Matterport 連携スクリプト
link to https://github.com/matterport/Mask_RCNN
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
import tkinter.scrolledtext
from tkinter import messagebox
import psutil
import threading
import time
import argparse
import socket
import configparser
import subprocess
from imgaug import augmenters as iaa
import shutil
import h5py
import keras

from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

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

sys.path.append(ROOT_DIR)
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
VAL_IMAGE_IDS = ["val"]
NumClass = 2

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
class CellTrackConfig(Config):
    global NumClass
    NAME = "cell"
    IMAGES_PER_GPU = 2#1#6
    NUM_CLASSES = NumClass # 1 + 1 #2
    STEPS_PER_EPOCH = -1 #
    EPOCH0 = 100
    EPOCH1 = 200
    VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)
    DETECTION_MIN_CONFIDENCE = 0.5
    BACKBONE = "resnet101"
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    POST_NMS_ROIS_TRAINING = 800
    POST_NMS_ROIS_INFERENCE = 2000
    RPN_NMS_THRESHOLD = 0.99
    RPN_TRAIN_ANCHORS_PER_IMAGE = 128
    MEAN_PIXEL = np.array([128,128,128])
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (102, 102)
    TRAIN_ROIS_PER_IMAGE = 256
    MAX_GT_INSTANCES = 500
    DETECTION_MAX_INSTANCES = 1000

class CellTrackInferenceConfig(CellTrackConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_RESIZE_MODE = "pad64"
    RPN_NMS_THRESHOLD = 0.7

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
class CellTrackDataset(utils.Dataset):
    global NumClass
    def load_cell_dataset(self, dataset_dir, subset):
        if NumClass > 2:
            for i in range(NumClass-1): #0,1
                self.add_class("cell", i+1, str(i+1))
        else:
            self.add_class("cell", 1, "Cells")

        if subset == "val":
            image_ids = VAL_IMAGE_IDS
        else:
            image_ids = next(os.walk(dataset_dir))[1]
            if subset == "train":
                image_ids = list(set(image_ids) - set(VAL_IMAGE_IDS))

        for image_id in image_ids:
            self.add_image("cell", image_id=image_id, path=os.path.join(dataset_dir, image_id, "images/{}.png".format(image_id)))

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")
        mask = []
        if NumClass > 2:
            class_ids = []
            for f in next(os.walk(mask_dir))[2]:
                if f.endswith(".png"):
                    tmp = skimage.io.imread(os.path.join(mask_dir, f))
                    tmpClassNo = tmp[tmp.nonzero()][0]
                    class_ids.append(tmpClassNo)
                    mask.append(tmp.astype(np.bool))

            mask = np.stack(mask, axis=-1)
            class_id_array = np.array(class_ids).astype(np.int32)
            return mask, class_id_array
        else:
            for f in next(os.walk(mask_dir))[2]:
                if f.endswith(".png"):
                    m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
                    mask.append(m)
            mask = np.stack(mask, axis=-1)
            return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "cell":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)

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
    global stop_flag
    global VAL_IMAGE_IDS
    global config
    global weightFolder
    global NumClass
    global model

    NumClass = int(args.numclass)

    weightFolder = " ----- "
    StartTime = datetime.datetime.today()
    #print('●StartTime:', StartTime)

    val_ratio = 0.05 #検証画像の割合（1.0未満）or枚数（1.0以上）
    _save_best_only = True
    _save_period = 1
    _original_setting = 0

    print()
    print("-----------------------------------------------------------");
    print(" Training start: ", StartTime)
    #print(" (Press CTRL+C to quit)")
    print("-----------------------------------------------------------");
    print()
    print("ValRatio: ", val_ratio)
    print("SaveBestOnly: ", _save_best_only)
    print("SavePeriod: ", _save_period)
    #print("---------------------------------------------------------")
    print("PostNmsRoisTraining: ", CellTrackConfig.POST_NMS_ROIS_TRAINING)
    #print("---------------------------------------------------------")
    print()

    #--------------------------------------------------------
    #検証用画像ファイルリスト VAL_IMAGE_IDS を設定する
    #valRatioが1未満の場合、全画像数＊valRatio枚ランダム選択した画像を検証画像設定
    #valRatioが1以上の場合、valRatio枚置きに抽出した画像を検証画像に設定
    image_ids = next(os.walk(args.dataset))[1] #ファイル名取得
    if val_ratio < 1:
        valCount = int(len(image_ids) * val_ratio)
        if valCount <= 1:       #ひとつの場合、最後の要素を選択
            VAL_IMAGE_IDS[0] = image_ids[-1]
        else:
            random.seed(0)      #複数個の場合、ランダム選択
            _VAL_IMAGE_IDS = random.choices(image_ids, k=valCount)
            VAL_IMAGE_IDS = _VAL_IMAGE_IDS[:]
            #print("VAL_IMAGE_IDS(val_ratio < 1 ): ", VAL_IMAGE_IDS)
    else:
        valStep = int(val_ratio)
        _VAL_IMAGE_IDS = image_ids[0:len(image_ids)-1:valStep] # [i for i in image_ids if i % valStep == 0] #valStep枚置きに抽出
        VAL_IMAGE_IDS = _VAL_IMAGE_IDS[:]
        #print("VAL_IMAGE_IDS(val_ratio >= 1 ): ", VAL_IMAGE_IDS)

    #--------------------------------------------------------

    argsstepperepoch = int(args.stepsperepoch)
    if argsstepperepoch < 1:
        argsstepperepoch = 1
    argsepoch0 = int(args.epoch0)
    if argsepoch0 < 1:
        argsepoch0 = 1
    argsepoch1 = int(args.epoch1)
    if argsepoch1 < 1:
        argsepoch1 = 1
    argstyperesnet = args.typeresnet
    if argstyperesnet!="resnet50" and argstyperesnet!="resnet101":
        argstyperesnet="resnet50"
    argsgpu = int(args.gpu)
    if argsgpu < 1:
        argsgpu = 1

    CellTrackConfig.STEPS_PER_EPOCH = argsstepperepoch
    CellTrackConfig.EPOCH0 = argsepoch0
    CellTrackConfig.EPOCH1 = argsepoch1
    CellTrackConfig.BACKBONE = argstyperesnet
    CellTrackConfig.IMAGES_PER_GPU = argsgpu
    CellTrackConfig.VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // CellTrackConfig.IMAGES_PER_GPU)
    print("StepsPerEpoch: ", CellTrackConfig.STEPS_PER_EPOCH)
    print("Epoch0: ", CellTrackConfig.EPOCH0)
    print("Epoch1: ", CellTrackConfig.EPOCH1)
    print("ImagesPerGPU: ", CellTrackConfig.IMAGES_PER_GPU)
    print("Backborn: ", CellTrackConfig.BACKBONE)

    if CellTrackConfig.BACKBONE == "resnet50" or _original_setting == 1:
        #print("Original parameter setting")
        CellTrackConfig.DETECTION_MIN_CONFIDENCE = 0
        CellTrackConfig.RPN_NMS_THRESHOLD = 0.9
        CellTrackConfig.RPN_TRAIN_ANCHORS_PER_IMAGE = 64
        CellTrackConfig.MEAN_PIXEL = np.array([43.53, 39.56, 48.22])
        CellTrackConfig.MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
        CellTrackConfig.TRAIN_ROIS_PER_IMAGE = 128
        CellTrackConfig.MAX_GT_INSTANCES = 200
        CellTrackConfig.DETECTION_MAX_INSTANCES = 1000

    config = CellTrackConfig()
    print()
    #print("Initial Weights: ", args.weights)
    print("OutputFolder: ", args.logs)
    if args.weights != "default":
        print("InitialWeight: ", args.weights)
    print("MaskImage: ", args.dataset)
    if NumClass > 2:
        print("NumClass: ", NumClass)

    print("ValImage: ", VAL_IMAGE_IDS)
    if os.name != 'nt':
        print()

    try:
        # Create model
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=os.path.join(args.logs, "__limtmp"))
    except Exception as e:
        messagebox.showerror('ERROR', 'Create model failed')
        if os.path.exists(os.path.join(args.logs, "__limtmp")):
              shutil.rmtree(os.path.join(args.logs, "__limtmp"))
        return

    #----------------------------------------
    #config.display()#パラメータを表示print()する場合
    #----------------------------------------

    if args.weights == "default":
        weights_path = ROOT_DIR + '/weights/resnet50_reduce.h5'
    else:
        weights_path = args.weights
        #print("Loading weight: ", weights_path)

    logdir = ""
    try:
        if NumClass > 2:
            #print("●train_service マルチクラス", NumClass)
            logdir = model.load_weights(weights_path, by_name=True, exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])#
        else:
            #print("●train_service シングルクラス", NumClass)
            logdir = model.load_weights(weights_path, by_name=True)#

    except Exception as e:
        if args.weights == "default":
            messagebox.showerror('ERROR', 'select failed')
        else:
            print("-----------------------------------------------------------");
            print(e)
            print("-----------------------------------------------------------");
            messagebox.showerror('ERROR', 'load_weights failed')
        if os.path.exists(os.path.join(args.logs, "__limtmp")):
              shutil.rmtree(os.path.join(args.logs, "__limtmp"))
        return

    now = datetime.datetime.now()
    weightFolder = os.path.join(args.logs, "weight_{:%Y%m%d%H%M}".format(now))
    checkpoint_path = os.path.join(weightFolder, "weight_{epoch:04d}.h5")

    threadTensorboard = threading.Thread(target=startTensorboard)
    threadTensorboard.start()

    dataset_train = CellTrackDataset()
    dataset_train.load_cell_dataset(args.dataset, "train")
    dataset_train.prepare()

    dataset_val = CellTrackDataset()
    dataset_val.load_cell_dataset(args.dataset, "val")
    dataset_val.prepare()

    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                    iaa.Affine(rotate=45),
                    iaa.Affine(rotate=45+90),
                    iaa.Affine(rotate=45+180),
                    iaa.Affine(rotate=45+270),
                    iaa.Affine(rotate=180),
                    iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # CustomCallbacks
    _callbacks = [
        keras.callbacks.TensorBoard(log_dir=weightFolder,
                                    histogram_freq=0,
                                    write_graph=True,
                                    write_images=False),
        keras.callbacks.ModelCheckpoint(checkpoint_path,
                                    monitor='loss',#
                                    verbose=0,
                                    save_best_only=_save_best_only,#
                                    save_weights_only=True,
                                    mode='min',#
                                    period=_save_period),#
    ]

    try:
        if config.EPOCH0 != 0:
            model.train(dataset_train, dataset_val,
                        learning_rate=config.LEARNING_RATE, #0.01
                        epochs=config.EPOCH0,
                        augmentation=augmentation,
                        layers='heads',
                        custom_callbacks=_callbacks)

        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=config.EPOCH1,
                    augmentation=augmentation,
                    layers='all',
                    custom_callbacks=_callbacks)

    except Exception as e:
        print(e)
        messagebox.showerror('ERROR', e)
        print("ERROR train_service failed")
        #messagebox.showerror('ERROR', 'model.train failed')
        if os.path.exists(os.path.join(args.logs, "__limtmp")):
              shutil.rmtree(os.path.join(args.logs, "__limtmp"))
        return

    if os.path.exists(os.path.join(args.logs, "__limtmp")):
          shutil.rmtree(os.path.join(args.logs, "__limtmp"))

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
def detect_service():
    global waitSocket
    global stop_flag
    global aftStart
    global titleLabel
    global config
    global weightFolder
    global NumClass
    global model

    NumClass = int(args.numclass)

    StartTime = datetime.datetime.today()
    #print('●StartTime:', StartTime)
    print()
    print("-----------------------------------------------------------");
    print(" Detection service start: ", StartTime)
    #print(" (Press CTRL+C to quit)")
    print("-----------------------------------------------------------");
    #print("---------------------------------------------------------")
    print("POST_NMS_ROIS_INFERENCE: ", CellTrackConfig.POST_NMS_ROIS_INFERENCE)
    #print("---------------------------------------------------------")
    print()
    #print("Weights: ", args.weights)

    weights_path = args.weights
    resnetKeysize = load_weights_length(weights_path)#
    if resnetKeysize != 407:
        #print("Resnet50 parameter setting")
        CellTrackConfig.BACKBONE = "resnet50"
        CellTrackConfig.DETECTION_MIN_CONFIDENCE = 0
        CellTrackConfig.RPN_TRAIN_ANCHORS_PER_IMAGE = 64
        CellTrackConfig.MEAN_PIXEL = np.array([43.53, 39.56, 48.22])
        CellTrackConfig.MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
        CellTrackConfig.TRAIN_ROIS_PER_IMAGE = 128
        CellTrackConfig.MAX_GT_INSTANCES = 200
        CellTrackConfig.DETECTION_MAX_INSTANCES = 1000
    config = CellTrackInferenceConfig()

    # Create model
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)

    #----------------------------------------
    #config.display()#パラメータを表示print()する場合
    #----------------------------------------

    if args.weights == "default":
        weights_path = ROOT_DIR + '/weights/resnet50_reduce.h5'
        #print("Loading weight: ", weights_path)
    else:
        weights_path = args.weights
        #print("Loading weight: ", weights_path)

    try:
        model.load_weights(weights_path, by_name=True)#

    except Exception as e:
        if args.weights == "default":
            messagebox.showerror('ERROR', 'select failed')
        else:
            print("-----------------------------------------------------------");
            print(e)
            print("-----------------------------------------------------------");
            messagebox.showerror('ERROR', 'load_weights failed')
        return

    ctx = zmq.Context()
    responder = ctx.socket(zmq.REP)
    responder.bind("tcp://*:11000")

    while not stop_flag:
        row = 0;
        col = 0;
        try:
            print()
            print("  READY!")
            print()
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

                image_sk1 = []
                image_sk1.append(image_cv)
                image_sk2 = model.detect(image_sk1)#, verbose=0)
                print("  model.detect: size=" + str(col) + "/" + str(row) + ", mat_type=" + str(mat_type)) #追加
                if len(image_sk2) != 0:
                    boxes = image_sk2[0]['rois']
                    masks = image_sk2[0]['masks']

                    if NumClass > 2:
                        classIDs = image_sk2[0]['class_ids']

                    roiCount = boxes.shape[0]
                    if not roiCount:
                        print("\n* No Roi * \n")
                    else:
                        assert boxes.shape[0] == masks.shape[-1]

                    masked_image = np.zeros((row, col), np.uint16)
                    if NumClass > 2:
                        #--------------------------------------------------------
                        class_image = np.zeros((row, col), np.uint16) # クラス画像出力用バッファ作成
                        #--------------------------------------------------------

                    #Create Mask Image
                    for roiNo in range(roiCount):
                        y1, x1, y2, x2 = boxes[roiNo]
                        mask = masks[y1:y2+1,x1:x2+1, roiNo] # mask = masks[:, :, roiNo]
                        masked_image[y1:y2+1,x1:x2+1] = np.where(mask == 1, (roiNo+1), masked_image[y1:y2+1,x1:x2+1])

                        if NumClass > 2:
                            #--------------------------------------------------------
                            classID = classIDs[roiNo] # mask = masks[:, :, roiNo]
                            class_image[y1:y2+1,x1:x2+1] = np.where(mask == 1, (classID), class_image[y1:y2+1,x1:x2+1])
                            #--------------------------------------------------------

                    if NumClass > 2:
                        data = [ np.array( [row] ), np.array( [col] ), np.array( [1] ), masked_image.data, class_image.data]
                    else:
                        data = [ np.array( [row] ), np.array( [col] ), np.array( [1] ), masked_image.data ]

                    responder.send_multipart(data)

        except Exception as e:
            print(e)
            messagebox.showerror('ERROR', e)
            print("ERROR detect_service failed")
            #messagebox.showerror('ERROR', 'detect_service failed')
            # ダミーデータを依頼側に送信
            masked_image = np.zeros((row, col), np.uint16) # マスク画像出力用バッファ作成
            if NumClass > 2:
                #--------------------------------------------------------
                class_image = np.zeros((row, col), np.uint16) # クラス画像出力用バッファ作成
                #--------------------------------------------------------

            if NumClass > 2:
                data = [ np.array( [row] ), np.array( [col] ), np.array( [1] ), masked_image.data, class_image.data ]
            else:
                data = [ np.array( [row] ), np.array( [col] ), np.array( [1] ), masked_image.data ]

            responder.send_multipart(data)
            print("ERROR detect_service failed")

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
def load_weights_length(filepath):
    f = h5py.File(filepath, mode='r')
    return len(f.keys()) #resnet50 = 237, resnet101 = 407

def mainFunction():
    global stop_flag
    global weightFolder
    global model
    global processType

    if processType == "train":
        #print("train mode: ")
        train_service()
        #os.system(weightFolder[0])
        EndTime = datetime.datetime.today()
        print()
        print("-----------------------------------------------------------");
        print(" Training end: ", EndTime)
        print("   --logdir= ", weightFolder)
        print("-----------------------------------------------------------");
        #wait = input(" Press ENTER to quit")
        #if os.name == 'nt':
        messagebox.showinfo('', 'Training process Completed')

    elif processType == "detect":
        #print("detect mode: ")
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
    if os.path.exists(os.path.join(args.logs, "__limtmp")):
          shutil.rmtree(os.path.join(args.logs, "__limtmp"))
    stop()

    #print("ExitProgramButton Clicked!")

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
        #1
        for p in psutil.process_iter(attrs=('name', 'pid', 'cmdline')):
            if 'python' in p.info['name'] and 'celltrack_service' in ' '.join(p.info['cmdline']):
                p = psutil.Process(p.info["pid"])
                p.kill()
        #2
        for p in psutil.process_iter(attrs=('name', 'pid', 'cmdline')):
            if 'python' in p.info['name'] and 'celltrack_service' in ' '.join(p.info['cmdline']):
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
        self.parent.title(" Matterport: LIM Track Service 20211129")

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

    if os.name == 'nt':
        UNIQUE_MUTEX_NAME = 'Global\\ProgramIsAlreadyRunning'
        handle = win32event.CreateMutex(None, pywintypes.FALSE, UNIQUE_MUTEX_NAME)
        if not handle or win32api.GetLastError() == winerror.ERROR_ALREADY_EXISTS:
            messagebox.showerror('ERROR', 'Another process is already running.')
            print('Another process is already running.', file=sys.stderr)
            sys.exit(-1)

    parser = argparse.ArgumentParser(description='Mask RCNN')
    parser.add_argument("command", metavar="<command>", help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False, metavar="/path/to/dataset/", help='Root directory of the dataset')
    parser.add_argument('--weights', required=True, metavar="/path/to/weights.h5", help="Path to weights .h5 file")
    parser.add_argument('--logs', required=False, default=DEFAULT_LOGS_DIR, metavar="/path/to/logs/", help='Logs')
    parser.add_argument('--stepsperepoch', required=False, metavar="Steps per EPOCH", help="Steps per EPOCH")
    parser.add_argument('--typeresnet', required=False, metavar="Resnet50 or Resnet101", help="Resnet50 or Resnet101")
    parser.add_argument('--epoch0', required=False, metavar="EPOCH0", help="EPOCH0")
    parser.add_argument('--epoch1', required=False, metavar="EPOCH1", help="EPOCH1")
    parser.add_argument('--gpu', required=False, metavar="IMAGES_PER_GPU", help="IMAGES_PER_GPU")
    parser.add_argument('--numclass', required=False, metavar="Number of Class", help="Number of Class")
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
