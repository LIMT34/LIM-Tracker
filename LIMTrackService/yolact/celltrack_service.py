"""
YOLACT++ 連携スクリプト
link to https://github.com/dbolya/yolact
"""
# coding: utf-8

import os
os.environ["PYTORCH_JIT"] = "0"

from data import *
from utils.augmentations import SSDAugmentation, BaseTransform, BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, SavePath, MovingAverage, ProgressBar
from utils.logger import Log
from utils import timer
from layers.modules import MultiBoxLoss
from layers.box_utils import jaccard, center_size, mask_iou
from layers.output_utils import postprocess, undo_image_transformation
from yolact import Yolact

import sys
import time
import math, random
from pathlib import Path
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as tdata
import numpy as np
import argparse
import datetime
import configparser
import eval as eval_script
import pycocotools
import cProfile
import pickle
import json
from collections import defaultdict
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import tkinter as tk
from tkinter import messagebox
import tkinter.simpledialog as simpledialog
from tkinter import messagebox
from tkinter import *
import tkinter.scrolledtext
import threading
import zmq
import struct
import psutil
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

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#学習用
parser = argparse.ArgumentParser(description='Yolact Training Script')

parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size for training')

parser.add_argument('--weights', default=None, type=str,
                    help='Checkpoint state_dict file to weights training from. If this is "interrupt"'\
                         ', the model will weights training from the interrupt file.')

parser.add_argument('--start_iter', default=-1, type=int,
                    help='Resume training at this iter. If this is -1, the iteration will be determined from the file name.')
                    #このイタレーションで学習再開。-1の場合は、ファイル名から繰り返し決定。

parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning_rate', default=None, type=float,
                    help='Initial learning rate. Leave as None to read this from the config.')
parser.add_argument('--momentum', default=None, type=float,
                    help='Momentum for SGD. Leave as None to read this from the config.')
parser.add_argument('--decay', '--weight_decay', default=None, type=float,
                    help='Weight decay for SGD. Leave as None to read this from the config.')
parser.add_argument('--gamma', default=None, type=float,
                    help='For each lr step, what to multiply the lr by. Leave as None to read this from the config.')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models.')
parser.add_argument('--log_folder', default='./logs/',
                    help='Directory for saving logs.')
parser.add_argument('--config', default=None,
                    help='The config object to use.')
parser.add_argument('--save_interval', default=10000, type=int,
                    help='The number of iterations between saving the model.')
parser.add_argument('--validation_size', default=5000, type=int,
                    help='The number of images to use for validation.')
parser.add_argument('--validation_epoch', default=2, type=int,
                    help='Output validation information every n iterations. If -1, do no validation.')
parser.add_argument('--keep_latest', dest='keep_latest', action='store_true',
                    help='Only keep the latest checkpoint instead of each one.')
parser.add_argument('--keep_latest_interval', default=100000, type=int,
                    help='When --keep_latest is on, don\'t delete the latest file at these intervals. This should be a multiple of save_interval or 0.')
parser.add_argument('--dataset', default=None, type=str,
                    help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')#configに設定されているデータセットを切り替える
parser.add_argument('--no_log', dest='log', action='store_false',
                    help='Don\'t log per iteration information into log_folder.')
parser.add_argument('--log_gpu', dest='log_gpu', action='store_true',
                    help='Include GPU information in the logs. Nvidia-smi tends to be slow, so set this with caution.')
parser.add_argument('--no_interrupt', dest='interrupt', action='store_false',
                    help='Don\'t save an interrupt when KeyboardInterrupt is caught.')
parser.add_argument('--batch_alloc', default=None, type=str,
                    help='If using multiple GPUS, you can set this to be a comma separated list detailing which GPUs should get what local batch size (It should add up to your total batch size).')
parser.add_argument('--no_autoscale', dest='autoscale', action='store_false',
                    help='YOLACT will automatically scale the lr and the number of iterations depending on the batch size. Set this if you want to disable that.')

parser.add_argument('--numclass', default=3, type=int, #背景＋Nuc＋Cyto
                    help='number of classes')

parser.add_argument('--dataset_folder', default=None, type=str,
                    help='Directory for saving checkpoint models.')

#認識用
parser.add_argument('--trained_model',
                    default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                    help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
parser.add_argument('--top_k', default=100, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--score_threshold', default=0, type=float,
                    help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
parser.add_argument('--image', default=None, type=str,
                    help='A path to an image to use for display.')
parser.add_argument('--display_lincomb', default=False, type=str2bool,
                    help='If the config uses lincomb masks, output a visualization of how those masks are created.')
parser.add_argument('--fast_nms', default=True, type=str2bool,
                    help='Whether to use a faster, but not entirely correct version of NMS.')
parser.add_argument('--cross_class_nms', default=False, type=str2bool,
                    help='Whether compute NMS cross-class or per-class.')
parser.add_argument('--display_masks', default=True, type=str2bool,
                    help='Whether or not to display masks over bounding boxes')
parser.add_argument('--display_bboxes', default=True, type=str2bool,
                    help='Whether or not to display bboxes around masks')
parser.add_argument('--display_text', default=True, type=str2bool,
                    help='Whether or not to display text (class [score])')
parser.add_argument('--display_scores', default=True, type=str2bool,
                    help='Whether or not to display scores in addition to classes')
parser.add_argument('--display', dest='display', action='store_true',
                    help='Display qualitative results instead of quantitative ones.')

parser.add_argument('--max_images', default=-1, type=int,#学習時の検証に必要
                    help='The maximum number of images from the dataset to consider. Use -1 for all.')
parser.add_argument('--no_bar', dest='no_bar', action='store_true',#学習時の検証に必要
                    help='Do not output the status bar. This is useful for when piping to a file.')

parser.add_argument('--mode', default='train', type=str, help='train or detect')

parser.set_defaults(keep_latest=False, log=True, log_gpu=False, interrupt=True, autoscale=True,
                    display_fps=False, crop=True, mask_proto_debug=False, no_bar=False, display=False)

args = parser.parse_args()

color_cache = defaultdict(lambda: {})

if args.config is not None:
    set_cfg(args.config)

if args.dataset is not None:
    set_dataset(args.dataset)

if args.autoscale and args.batch_size != 8:
    factor = args.batch_size / 8
    if __name__ == '__main__':
        print('Scaling parameters by %.2f to account for a batch size of %d.' % (factor, args.batch_size))

    cfg.lr *= factor
    cfg.max_iter //= factor
    cfg.lr_steps = [x // factor for x in cfg.lr_steps]

# Update training parameters from the config if necessary
def replace(name):
    if getattr(args, name) == None: setattr(args, name, getattr(cfg, name))
replace('lr')
replace('decay')
replace('gamma')
replace('momentum')

# This is managed by set_lr
cur_lr = args.lr

if torch.cuda.device_count() == 0:
    print('No GPUs detected. Exiting...')
    exit(-1)

if args.batch_size // torch.cuda.device_count() < 6:
    if __name__ == '__main__':
        print('Per-GPU batch size is less than the recommended limit for batch norm. Disabling batch norm.')
    cfg.freeze_bn = True

loss_types = ['B', 'C', 'M', 'P', 'D', 'E', 'S', 'I']

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

class NetLoss(nn.Module):
    """
    A wrapper for running the network and computing the loss
    This is so we can more efficiently use DataParallel.
    """

    def __init__(self, net:Yolact, criterion:MultiBoxLoss):
        super().__init__()

        self.net = net
        self.criterion = criterion

    def forward(self, images, targets, masks, num_crowds):
        preds = self.net(images)
        losses = self.criterion(self.net, preds, targets, masks, num_crowds)
        return losses

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
class CustomDataParallel(nn.DataParallel):
    """
    This is a custom version of DataParallel that works better with our training data.
    It should also be faster than the general case.
    """

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:' + str(x) for x in device_ids]
        splits = prepare_data(inputs[0], devices, allocation=args.batch_alloc)

        return [[split[device_idx] for split in splits] for device_idx in range(len(devices))], \
            [kwargs] * len(devices)

    def gather(self, outputs, output_device):
        out = {}

        for k in outputs[0]:
            out[k] = torch.stack([output[k].to(output_device) for output in outputs])

        return out

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
def train_service():

    #データセットフォルダ
    if args.dataset_folder is not None:
        if not os.path.exists(args.dataset_folder):
            print("指定されたデータセットフォルダがない")
            return
        else:
            print("Dataset Folder: ", args.dataset_folder)
            cfg.dataset.train_images = args.dataset_folder
            cfg.dataset.train_info = args.dataset_folder + "annotations.json"
            cfg.dataset.valid_images = args.dataset_folder
            cfg.dataset.valid_info = args.dataset_folder + "annotations.json"

    #クラス名
    if args.numclass == 3:
        cfg.dataset.class_names = ('Nuc', 'Cell')
    else:
        cfg.dataset.class_names = ('Cell')

    #重みセーブフォルダ
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    try:
        #データセット作成
        dataset = COCODetection(image_path=cfg.dataset.train_images,
                                info_file=cfg.dataset.train_info,
                                transform=SSDAugmentation(MEANS))
    except Exception as e:
        messagebox.showerror('ERROR1', str(e) + "\n\n Adjust the Parameter -> Mask image.")
        exit()

    #バッチサイズ確認
    if args.batch_size > len(dataset):
        args.batch_size = len(dataset)

    if args.validation_epoch > 0:
        setup_eval()
        val_dataset = COCODetection(image_path=cfg.dataset.valid_images,
                                    info_file=cfg.dataset.valid_info,
                                    transform=BaseTransform(MEANS))




    # Parallel wraps the underlying module, but when saving and loading we don't want that
    yolact_net = Yolact()
    net = yolact_net
    net.train()

    if args.log:
        log = Log(cfg.name, args.save_folder, dict(args._get_kwargs()),
            overwrite=(args.weights is None), log_gpu_stats=args.log_gpu)

    # I don't use the timer during training (I use a different timing method).
    # Apparently there's a race condition with multiple GPUs, so disable it just to be safe.
    timer.disable_all()

    # Both of these can set args.weights to None, so do them before the check
    if args.weights == 'interrupt':
        args.weights = SavePath.get_interrupt(args.save_folder)
    elif args.weights == 'latest':
        args.weights = SavePath.get_latest(args.save_folder, cfg.name)

    if args.weights is not None and args.weights != 'default':
        print('Resuming training, loading {}...'.format(args.weights))
        yolact_net.load_weights(args.weights)





        if args.start_iter == -1:
            args.start_iter = SavePath.from_str(args.weights).iteration
    else:
        print('Initializing weights...')

        inifile = configparser.ConfigParser()
        # determine if application is a script file or frozen exe
        if getattr(sys, 'frozen', False):
            application_path = os.path.dirname(sys.executable)
        elif __file__:
            application_path = os.path.dirname(__file__)

        #yolact_net.init_weights(backbone_path=args.save_folder + cfg.backbone.path)
        yolact_net.init_weights(backbone_path=application_path + "./weights/" + cfg.backbone.path)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)
    criterion = MultiBoxLoss(num_classes=cfg.num_classes,
                             pos_threshold=cfg.positive_iou_threshold,
                             neg_threshold=cfg.negative_iou_threshold,
                             negpos_ratio=cfg.ohem_negpos_ratio)

    if args.batch_alloc is not None:
        args.batch_alloc = [int(x) for x in args.batch_alloc.split(',')]
        if sum(args.batch_alloc) != args.batch_size:
            print('Error: Batch allocation (%s) does not sum to batch size (%s).' % (args.batch_alloc, args.batch_size))
            exit(-1)







    net = CustomDataParallel(NetLoss(net, criterion))
    if args.cuda:
        net = net.cuda()

    # Initialize everything
    if not cfg.freeze_bn: yolact_net.freeze_bn() # Freeze bn so we don't kill our means
    yolact_net(torch.zeros(1, 3, cfg.max_size, cfg.max_size).cuda())
    if not cfg.freeze_bn: yolact_net.freeze_bn(True)

    print()
    # print("------------------------------------------")
    # SSD data augmentation parameters
    # Randomize hue, vibrance, etc.
    print("augment_photometric_distort: ", cfg.augment_photometric_distort)#': True,
    # Have a chance to scale down the image and pad (to emulate smaller detections)
    print("augment_expand: ", cfg.augment_expand)#': True,
    # Potentialy sample a random crop from the image and put it in a random place
    print("augment_random_sample_crop: ", cfg.augment_random_sample_crop)#': True,
    # Mirror the image with a probability of 1/2 1/2
    print("augment_random_mirror: ", cfg.augment_random_mirror)#': True,
    # Flip the image vertically with a probability of 1/2
    print("augment_random_flip: ", cfg.augment_random_flip)#': False,
    # With uniform probability, rotate the image [0,90,180,270] degrees
    print("augment_random_rot90: ", cfg.augment_random_rot90)#': False,
    #追加
    #print("augment_random_multiply: ", cfg.augment_random_multiply)#': False,
    #追加
    #print("augment_random_gaussianblur: ", cfg.augment_random_gaussianblur)#': False,

    print("train_images: ", cfg.dataset.train_images)
    print("train_info: ", cfg.dataset.train_info)
    print("valid_images: ", cfg.dataset.valid_images)
    print("valid_info: ", cfg.dataset.valid_info)

    print("save_folder: ", args.save_folder)
    print("weights: ", args.weights)
    print("class_names: ", cfg.dataset.class_names)
    print("num_classes: ", cfg.num_classes)
    print("max_size: ", cfg.max_size)
    print("batch_size: ", args.batch_size)
    print("lr_steps: ", cfg.lr_steps)
    print("max_iter: ", cfg.max_iter)
    print("pred_aspect_ratios: ", cfg.backbone.pred_aspect_ratios)
    print("pred_scales: ", cfg.backbone.pred_scales)
    # print("------------------------------------------")
    print()


    # loss counters
    loc_loss = 0
    conf_loss = 0
    iteration = max(args.start_iter, 0)
    last_time = time.time()

    epoch_size = len(dataset) // args.batch_size
    num_epochs = math.ceil(cfg.max_iter / epoch_size)

    # Which learning rate adjustment step are we on? lr' = lr * gamma ^ step_index
    step_index = 0

    data_loader = tdata.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)


    save_path = lambda epoch, iteration: SavePath(cfg.name, epoch, iteration).get_path(root=args.save_folder)
    time_avg = MovingAverage()

    global loss_types # Forms the print order
    loss_avgs  = { k: MovingAverage(100) for k in loss_types }

    print('Begin training!')
    print()
    # try-except so you can use ctrl+c to save early and stop training
    try:
        for epoch in range(num_epochs):
            # Resume from start_iter
            if (epoch+1)*epoch_size < iteration:
                continue

            for datum in data_loader:
                # Stop if we've reached an epoch if we're resuming from start_iter
                if iteration == (epoch+1)*epoch_size:
                    break

                # Stop at the configured number of iterations even if mid-epoch
                if iteration == cfg.max_iter:
                    break

                # Change a config setting if we've reached the specified iteration
                changed = False
                for change in cfg.delayed_settings:
                    if iteration >= change[0]:
                        changed = True
                        cfg.replace(change[1])

                        # Reset the loss averages because things might have changed
                        for avg in loss_avgs:
                            avg.reset()

                # If a config setting was changed, remove it from the list so we don't keep checking
                if changed:
                    cfg.delayed_settings = [x for x in cfg.delayed_settings if x[0] > iteration]

                # Warm up by linearly interpolating the learning rate from some smaller value
                if cfg.lr_warmup_until > 0 and iteration <= cfg.lr_warmup_until:
                    set_lr(optimizer, (args.lr - cfg.lr_warmup_init) * (iteration / cfg.lr_warmup_until) + cfg.lr_warmup_init)

                # Adjust the learning rate at the given iterations, but also if we weights from past that iteration
                while step_index < len(cfg.lr_steps) and iteration >= cfg.lr_steps[step_index]:
                    step_index += 1
                    set_lr(optimizer, args.lr * (args.gamma ** step_index))

                # Zero the grad to get ready to compute gradients
                optimizer.zero_grad()

                # Forward Pass + Compute loss at the same time (see CustomDataParallel and NetLoss)
                losses = net(datum)

                losses = { k: (v).mean() for k,v in losses.items() } # Mean here because Dataparallel
                loss = sum([losses[k] for k in losses])

                # no_inf_mean removes some components from the loss, so make sure to backward through all of it
                # all_loss = sum([v.mean() for v in losses.values()])

                # Backprop
                loss.backward() # Do this to free up vram even if loss is not finite
                if torch.isfinite(loss).item():
                    optimizer.step()

                # Add the loss to the moving average for bookkeeping
                for k in losses:
                    loss_avgs[k].add(losses[k].item())

                cur_time  = time.time()
                elapsed   = cur_time - last_time
                last_time = cur_time

                # Exclude graph setup from the timing information
                if iteration != args.start_iter:
                    time_avg.add(elapsed)

                if iteration % 10 == 0:
                    eta_str = str(datetime.timedelta(seconds=(cfg.max_iter-iteration) * time_avg.get_avg())).split('.')[0]

                    total = sum([loss_avgs[k].get_avg() for k in losses])
                    loss_labels = sum([[k, loss_avgs[k].get_avg()] for k in loss_types if k in losses], [])

                    print(('[%3d] %7d ||' + (' %s: %.3f |' * len(losses)) + ' T: %.3f || ETA: %s || timer: %.3f')
                            % tuple([epoch, iteration] + loss_labels + [total, eta_str, elapsed]), flush=True)

                if args.log:
                    precision = 5
                    loss_info = {k: round(losses[k].item(), precision) for k in losses}
                    loss_info['T'] = round(loss.item(), precision)

                    if args.log_gpu:
                        log.log_gpu_stats = (iteration % 10 == 0) # nvidia-smi is sloooow

                    log.log('train', loss=loss_info, epoch=epoch, iter=iteration,
                        lr=round(cur_lr, 10), elapsed=elapsed)

                    log.log_gpu_stats = args.log_gpu

                iteration += 1

                if iteration % args.save_interval == 0 and iteration != args.start_iter:
                    if args.keep_latest:
                        latest = SavePath.get_latest(args.save_folder, cfg.name)

                    print('Saving state, iter:', iteration)
                    yolact_net.save_weights(save_path(epoch, iteration))

                    if args.keep_latest and latest is not None:
                        if args.keep_latest_interval <= 0 or iteration % args.keep_latest_interval != args.save_interval:
                            print('Deleting old save...')
                            os.remove(latest)

            # This is done per epoch
            if args.validation_epoch > 0:
                if epoch % args.validation_epoch == 0 and epoch > 0:
                    compute_validation_map(epoch, iteration, yolact_net, val_dataset, log if args.log else None)

        # Compute validation mAP after training is finished
        compute_validation_map(epoch, iteration, yolact_net, val_dataset, log if args.log else None)

    except KeyboardInterrupt:
        if args.interrupt:
            print('Stopping early. Saving network...')

            # Delete previous copy of the interrupted network so we don't spam the weights folder
            SavePath.remove_interrupt(args.save_folder)

            yolact_net.save_weights(save_path(epoch, repr(iteration) + '_interrupt'))
        exit()

    except Exception as e:
        messagebox.showerror('ERROR3', e)
        print("-> dataset num_classes?")
        exit()

    yolact_net.save_weights(save_path(epoch, iteration))

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    global cur_lr
    cur_lr = new_lr

def gradinator(x):
    x.requires_grad = False
    return x

def prepare_data(datum, devices:list=None, allocation:list=None):
    with torch.no_grad():
        if devices is None:
            devices = ['cuda:0'] if args.cuda else ['cpu']
        if allocation is None:
            allocation = [args.batch_size // len(devices)] * (len(devices) - 1)
            allocation.append(args.batch_size - sum(allocation)) # The rest might need more/less

        images, (targets, masks, num_crowds) = datum

        cur_idx = 0
        for device, alloc in zip(devices, allocation):
            for _ in range(alloc):
                images[cur_idx]  = gradinator(images[cur_idx].to(device))
                targets[cur_idx] = gradinator(targets[cur_idx].to(device))
                masks[cur_idx]   = gradinator(masks[cur_idx].to(device))
                cur_idx += 1

        if cfg.preserve_aspect_ratio:
            # Choose a random size from the batch
            _, h, w = images[random.randint(0, len(images)-1)].size()

            for idx, (image, target, mask, num_crowd) in enumerate(zip(images, targets, masks, num_crowds)):
                images[idx], targets[idx], masks[idx], num_crowds[idx] \
                    = enforce_size(image, target, mask, num_crowd, w, h)

        cur_idx = 0
        split_images, split_targets, split_masks, split_numcrowds \
            = [[None for alloc in allocation] for _ in range(4)]

        for device_idx, alloc in enumerate(allocation):
            split_images[device_idx]    = torch.stack(images[cur_idx:cur_idx+alloc], dim=0)
            split_targets[device_idx]   = targets[cur_idx:cur_idx+alloc]
            split_masks[device_idx]     = masks[cur_idx:cur_idx+alloc]
            split_numcrowds[device_idx] = num_crowds[cur_idx:cur_idx+alloc]

            cur_idx += alloc

        return split_images, split_targets, split_masks, split_numcrowds

def no_inf_mean(x:torch.Tensor):
    """
    Computes the mean of a vector, throwing out all inf values.
    If there are no non-inf values, this will return inf (i.e., just the normal mean).
    """

    no_inf = [a for a in x if torch.isfinite(a)]

    if len(no_inf) > 0:
        return sum(no_inf) / len(no_inf)
    else:
        return x.mean()

def compute_validation_loss(net, data_loader, criterion):
    global loss_types

    with torch.no_grad():
        losses = {}

        # Don't switch to eval mode because we want to get losses
        iterations = 0
        for datum in data_loader:
            images, targets, masks, num_crowds = prepare_data(datum)
            out = net(images)

            wrapper = ScatterWrapper(targets, masks, num_crowds)
            _losses = criterion(out, wrapper, wrapper.make_mask())

            for k, v in _losses.items():
                v = v.mean().item()
                if k in losses:
                    losses[k] += v
                else:
                    losses[k] = v

            iterations += 1
            if args.validation_size <= iterations * args.batch_size:
                break

        for k in losses:
            losses[k] /= iterations


        loss_labels = sum([[k, losses[k]] for k in loss_types if k in losses], [])
        print(('Validation ||' + (' %s: %.3f |' * len(losses)) + ')') % tuple(loss_labels), flush=True)

def compute_validation_map(epoch, iteration, yolact_net, dataset, log:Log=None):
    #print("compute_validation_mapは実行しない！")
    with torch.no_grad():
        yolact_net.eval()

        start = time.time()
        print()
        print("Computing validation mAP (this may take a while)...", flush=True)
        val_info = eval_script.evaluate(yolact_net, dataset, train_mode=True)
        end = time.time()

        if log is not None:
            log.log('val', val_info, elapsed=(end - start), epoch=epoch, iter=iteration)

        yolact_net.train()

def setup_eval():
    #print("eval_script.parse_argsは実行しない！")
    eval_script.parse_args(['--no_bar', '--max_images='+str(args.validation_size)])


#認識用
def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape

    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(dets_out, w, h, visualize_lincomb = args.display_lincomb,
                                        crop_masks        = args.crop,
                                        score_threshold   = args.score_threshold)
        cfg.rescore_bbox = save

    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:args.top_k]

        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

    num_dets_to_consider = min(args.top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < args.score_threshold:
            num_dets_to_consider = j
            break

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)

        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if args.display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]

        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1

        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

    if args.display_fps:
            # Draw the box for the fps on the GPU
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        text_w, text_h = cv2.getTextSize(fps_str, font_face, font_scale, font_thickness)[0]

        img_gpu[0:text_h+8, 0:text_w+8] *= 0.6 # 1 - Box alpha


    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()

    if args.display_fps:
        # Draw the text on the CPU
        text_pt = (4, text_h + 2)
        text_color = [255, 255, 255]

        cv2.putText(img_numpy, fps_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

    if num_dets_to_consider == 0:
        return img_numpy

    if args.display_text or args.display_bboxes:
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j)
            score = scores[j]

            if args.display_bboxes:
                cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

            if args.display_text:
                _class = cfg.dataset.class_names[classes[j]]
                text_str = '%s: %.2f' % (_class, score) if args.display_scores else _class

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.4 #0.6
                font_thickness = 1

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]

                #cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)


    return img_numpy

#認識用
def predict_instances(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):

    img_gpu = img / 255.0
    h, w, _ = img.shape

    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(dets_out, w, h, visualize_lincomb = args.display_lincomb,
                                        crop_masks        = args.crop,
                                        score_threshold   = args.score_threshold)
        cfg.rescore_bbox = save

    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:args.top_k]

        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]
        #print("boxes", boxes)

    num_dets_to_consider = min(args.top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < args.score_threshold:
            num_dets_to_consider = j
            break

    def get_color(j, on_gpu=None):
        color = COLORS[0]
        color = (1, 1, 1)
        color = torch.Tensor(color).to(on_gpu).float()
        return color

    #np.set_printoptions(threshold=numpy.inf)

    if num_dets_to_consider > 0:
        masks = masks[:num_dets_to_consider, :, :, None]
        colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors #各マスク情報

        numbuff = masks_color[0].byte().cpu().numpy()
        for c in range(3):
            numbuff[:, :, c] = 0
        maskImage = masks_color.byte().cpu().numpy()
        segNo = 1
        for roiNo in range(num_dets_to_consider):


            x1, y1, x2, y2 = boxes[roiNo]
            if np.any(maskImage[roiNo, y1:y2+1,x1:x2+1, 1] == 1): #領域が含まれていない場合もあるので追加
                valR, amari = divmod((segNo), 256*256)
                valG, valB = divmod(amari, 256)

                # np.set_printoptions(linewidth=1000)
                # np.set_printoptions(threshold=100000)
                # print("maskImage", maskImage[roiNo, y1:y2+1,x1:x2+1, 1])

                numbuff[y1:y2+1,x1:x2+1, 0] = np.where(maskImage[roiNo, y1:y2+1,x1:x2+1, 0] == 1, (valR), numbuff[y1:y2+1,x1:x2+1, 0])
                numbuff[y1:y2+1,x1:x2+1, 1] = np.where(maskImage[roiNo, y1:y2+1,x1:x2+1, 1] == 1, (valG), numbuff[y1:y2+1,x1:x2+1, 1])
                numbuff[y1:y2+1,x1:x2+1, 2] = np.where(maskImage[roiNo, y1:y2+1,x1:x2+1, 2] == 1, (valB), numbuff[y1:y2+1,x1:x2+1, 2])
                segNo = segNo+1

        return  numbuff

    return None

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
def detect_service(argv=None):
    global waitSocket
    global stop_flag
    global aftStart
    global titleLabel

    if args.config is not None:
        set_cfg(args.config)

    if args.config is None:
        try:
            model_path = SavePath.from_str(args.trained_model)
            # TODO: Bad practice? Probably want to do a name lookup instead.
            args.config = model_path.model_name + '_config'
            print('Config not specified. Parsed %s from the file name.\n' % args.config)
            set_cfg(args.config)

        except Exception as e:
            messagebox.showerror('ERROR', 'detect_service SavePath.from_str failed')
            print("ERROR detect_service SavePath.from_str failed")
            stop()

    with torch.no_grad():
        try:
            if args.cuda:
                cudnn.fastest = True
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            else:
                torch.set_default_tensor_type('torch.FloatTensor')

            dataset = None
            print("trained_model: ", args.trained_model)
            print('Loading model...', end='')
            net = Yolact()
            net.load_weights(args.trained_model)
            net.eval()
            print(' Done.')

            if args.cuda:
                net = net.cuda()

            net.detect.use_fast_nms = args.fast_nms
            net.detect.use_cross_class_nms = args.cross_class_nms
            cfg.mask_proto_debug = args.mask_proto_debug

            cfg.max_num_detections =  args.top_k
            cfg.nms_top_k =  args.top_k*2

            ctx = zmq.Context()
            responder = ctx.socket(zmq.REP)
            responder.bind("tcp://*:11000")

        except Exception as e:
            messagebox.showerror('ERROR', 'detect_service load_weights failed')
            print("ERROR detect_service load_weights failed")
            stop()

        while not stop_flag:
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
                    row = struct.unpack('i', byte_rows)
                    col = struct.unpack('i', byte_cols)
                    mat_type = struct.unpack('i', byte_mat_type)

                    if mat_type[0] != 3:
                        # Gray Scale
                        image_cv = np.frombuffer(data, dtype=np.uint8).reshape((row[0],col[0]));
                    else:
                        # BGR -> RGB Color
                        image_cv = np.frombuffer(data, dtype=np.uint8).reshape((row[0],col[0],3));
                        image_cv = image_cv[:, :, [2, 1, 0]]
                        #cv2.imwrite("C:/CellTrackService_yolact/test_result_img/test_in.png", image_cv)

                    frame = torch.from_numpy(image_cv).cuda().float()
                    batch = FastBaseTransform()(frame.unsqueeze(0))
                    preds = net(batch)
                    masked_image = predict_instances(preds, frame, None, None, undo_transform=False)
                    print("  model.predict_instances: size=" + str(col) + "/" + str(row) + ", mat_type=" + str(mat_type)) #追加
                    #outFile = "C:/CellTrackService_yolact/test_result_img/test_out.png"
                    #cv2.imwrite(outFile, masked_image)
                    if masked_image is None:
                        masked_image = image_cv.astype(np.uint8).copy()
                        for c in range(3):
                            masked_image[:, :, c] = 0

                    height, width = masked_image.shape[:2]
                    ndim = masked_image.ndim
                    data = [ np.array( [height] ), np.array( [width] ), np.array( [ndim] ), masked_image.data ]
                    responder.send_multipart(data)

            except Exception as e:
                messagebox.showerror('ERROR', 'detect_service failed')
                # ダミーデータ送信
                masked_image = image_cv.astype(np.uint8).copy()
                height, width = masked_image.shape[:2]
                ndim = masked_image.ndim
                for c in range(3):
                    masked_image[:, :, c] = 0

                if NumClass > 2:
                    # クラス画像出力用バッファ作成
                    class_image = image_cv.astype(np.uint8).copy()
                    ndim = class_image.ndim
                    for c in range(3):
                        class_image[:, :, c] = 0

                if NumClass > 2:
                    data = [ np.array( [height] ), np.array( [width] ), np.array( [ndim] ), masked_image.data, class_image.data]
                else:
                    data = [ np.array( [height] ), np.array( [width] ), np.array( [ndim] ), masked_image.data ]

                responder.send_multipart(data)
                print("ERROR detect_service failed")

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
def mainFunction():
    global stop_flag

    if args.mode == "train":
        print("train mode: ")
        train_service()
        #os.system(weightFolder[0])
        EndTime = datetime.datetime.today()
        print()
        print("-----------------------------------------------------------");
        print(" Training end: ", EndTime)
        print("-----------------------------------------------------------");
        #wait = input(" Press ENTER to quit")
        if os.name == 'nt':
            messagebox.showinfo('', 'Training process Completed')

    elif args.mode == "detect":
        print("detect mode: ")
        detect_service()
    # elif args.mode == "eval":
    #     print("eval mode: ")
    #     detect()
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
            if proc.name() == "tensorboard.exe" or proc.name() == "train_detect_service_yolact_gui.exe":
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
class CoreGUI(object):
    def __init__(self, parent):
        global titleLabel
        global processType
        global closeButton

        self.parent = parent
        self.parent.title(" YOLACT++: LIM Track Service 20211129")

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
        UNIQUE_MUTEX_NAME = 'Global\\MyProgramIsAlreadyRunning'
        handle = win32event.CreateMutex(None, pywintypes.FALSE, UNIQUE_MUTEX_NAME)
        if not handle or win32api.GetLastError() == winerror.ERROR_ALREADY_EXISTS:
            messagebox.showerror('ERROR', 'Another process is already running.')
            print('Another process is already running.', file=sys.stderr)
            sys.exit(-1)

    processType = args.mode

    flg = True
    if(flg):
        if os.name == 'nt' or args.mode == "detect":
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

        else:
            mainFunction()
