import argparse
import json
import os
from pathlib import Path
from threading import Thread

import time
import numpy as np
import torch
import yaml
from tqdm import tqdm
import cv2

import matplotlib.pyplot as plt
import torch.nn as nn
from detect import detect
from models.experimental import attempt_load
from utils.datasets import create_dataloader, letterbox
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import math
from scipy import signal
from quant_utils import *

model_name = 'voc_retrain' # coco_origin | coco128_retrain | voc_retrain
relative_path = './models_files/' + model_name
source = 'data/images'
dataset = LoadImages('data/images', img_size=416, stride=32)

if(model_name == 'voc_retrain'):
    model_structure = './models/yolov3-tiny_voc.yaml'
    float_model_files = relative_path + '/train/weights/best.pt'
    quant_model_files = relative_path + '/yolov3tiny_quant.pth'
    class_num = 72
    prefix = 'V'
elif(model_name == 'coco_origin'):
    model_structure = './models/yolov3-tiny_voc.yaml'
    float_model_files = relative_path + '/train/weights/best.pt'
    quant_model_files = relative_path + '/yolov3tiny_quant.pth'
    class_num = 72
    prefix = 'Y'
elif(model_name == 'coco128_retrain'):
    model_structure = './models/yolov3-tiny_voc.yaml'
    float_model_files = relative_path + '/train/weights/best.pt'
    quant_model_files = relative_path + '/yolov3tiny_quant.pth'
    class_num = 72
    prefix = 'Y'
elif(model_name == 'coco_origin'):
    model_structure = './models/yolov3-tiny_voc.yaml'
    float_model_files = relative_path + '/train/weights/best.pt'
    quant_model_files = relative_path + '/yolov3tiny_quant.pth'
    class_num = 72
    prefix = 'Y'
#示例
# float_model,dataloader_iter=load_model_data('kaggle-facemask.yaml','yolov3tiny_facemask.pt',416,False)
# quant_model=load_quant_model('yolov3tiny_facemask.pt','yolov3tiny_facemask_quant.pth')

# float_model,dataloader_iter=load_model_data('yolov3-tiny-facemask.yaml','./models_files/yolov3tiny_facemask.pt',416,False)
# quant_model=load_quant_model('./models_files/yolov3tiny_facemask.pt','./yolov3tiny_facemask_quant.pth')
# yolov3tiny_infer_para_gen(quant_model,24,'F') 

# float_model,dataloader_iter=load_model_data('yolov3-tiny.yaml','models_files\yolov3-tiny.pt',416,False)
# quant_model=load_quant_model('models_files\yolov3-tiny.pt','./yolov3tiny_quant.pth')
# yolov3tiny_infer_para_gen(quant_model,32,'Y')

# float_model,dataloader_iter=load_model_data('yolov3-tiny.yaml','runs/train/exp13/weights/best.pt',416,False) # 训练后模型
# float_model,dataloader_iter=load_model_data('./models/yolov3-tiny.yaml',relative_path + '/best.pt',416,False) # 官方模型
# quant_model=load_quant_model(relative_path + '/best.pt', relative_path + '/yolov3tiny_quant.pth')
# yolov3tiny_infer_para_gen(quant_model,256,'Y',relative_path)


float_model,dataloader_iter=load_model_data(model_structure, float_model_files, 416, False) 
names = float_model.module.names if hasattr(float_model, 'module') else float_model.names  # get class names
model_quant_save(dataset, float_model, quant_model_files)
quant_model=load_quant_model(float_model_files, quant_model_files)
yolov3tiny_infer_para_gen(quant_model, class_num, prefix, relative_path, model_name)

#detect
# view_img = check_imshow()
# cudnn.benchmark = True  # set True to speed up constant image size inference
# dataset = LoadStreams('0', img_size=416, stride=32)
# state_dict = torch.load('./yolov3tiny_facemask_quant.pth')

# data = cv2.imread('data\images\zidane.jpg')
# data = cv2.imread('data/images/bus.jpg')
# data = cv2.imread('data/images/test_1.jpg')
# data = cv2.imread('data/images/test_1.jpg')


# data,rate,d = letterbox(data,(416, 416),32)
# # cv2.imshow('input_image', data)
# # cv2.waitKey(0) 
# data = np.expand_dims(data.transpose(2,0,1),axis = 0)
# data = np.expand_dims(data,axis = 0)
# data = torch.from_numpy(data)
# quant_model_evaluate_show(data,quant_model)
for path, img, im0s, vid_cap in dataset:
    # img=img.swapaxes(0,1)
    # img=img.swapaxes(1,2)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # img=img.swapaxes(1,2)
    # img=img.swapaxes(0,1)
    img = torch.from_numpy(img).to('cpu')
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    quant_model_evaluate_show_name(img,quant_model,im0s,names)
