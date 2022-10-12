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
# from yolov3tiny_quant import quant_model_evaluate_show

def quant_model_detect(x,quant_model):
    x=quant_model[0](x)
    for ii in range(11):
        print(ii,quant_model[1].model[ii])
        x=quant_model[1].model[ii](x)
    x_scale = x.q_scale()
    x_zq = x.q_zero_point()
    x_type = x.dtype
    x = x.dequantize()
    # print(x)
    p = torch.ones(x.shape[2]*x.shape[1]) * 0
    # print(p)
    p = p.reshape(x.shape[0],x.shape[1],x.shape[2],1)
    # # print(x)
    # print(p)
    x = torch.cat((x,p),3)

    # print(x)

    q = torch.ones(x.shape[1]*x.shape[3]) * 0
    # q = q.unsqueeze(0)
    # print(q)
    q = q.reshape(x.shape[0],x.shape[1],1,x.shape[3])
    # print(q)
    x = torch.cat((x,q),2)
    x = torch.quantize_per_tensor(x, scale = x_scale, zero_point = x_zq, dtype=x_type)

    # print(x)
    # print(x.scale)
    for ii in range(12,16):
        print(ii,quant_model[1].model[ii])
        x=quant_model[1].model[ii](x)
    print(ii+1,quant_model[1].model[-1].m[1])
    x=quant_model[1].model[-1].m[1](x)
    x=quant_model[2](x)
    return x

def quant_model_evaluate_show(data,quant_model):
    na=quant_model[1].model[-1].na
    nl=quant_model[1].model[-1].nl
    no=quant_model[1].model[-1].no
    stride=quant_model[1].model[-1].stride
    anchor=quant_model[1].model[-1].anchors
    anchor_grid=quant_model[1].model[-1].anchor_grid

    # img=data[0]
    # x=(img.float()/255.0)
    x = data

    res=quant_model_detect(x,quant_model)
    pred_reduce=torch.dequantize(res)
    
    bs, _, ny, nx = pred_reduce.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
    y = pred_reduce.view(bs, na, no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

    y=y.sigmoid()
    grid=quant_model[1].model[-1]._make_grid(nx, ny)
    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid) * stride[1]  # xy
    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[1]  # wh
    y=y.view(1, -1, no)
    print(y)
    y_nms=non_max_suppression(y, 0.25, 0.5, None, False,max_det=1000)

    img_numpy=img[0,:,:,:].numpy()
    img_numpy=img_numpy.swapaxes(0,1)
    img_numpy=img_numpy.swapaxes(1,2)
    img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
    cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE)
    for ii in y_nms[0][:,:4].round():
        x1,y1,x2,y2=ii[0].int().item(),ii[1].int().item(),ii[2].int().item(),ii[3].int().item()
        cv2.rectangle(img_numpy, (x1,y1), (x2,y2), (0, 0, 255), 2)
    cv2.imshow('input_image', img_numpy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def load_model_data(data,weights,imgsz,rect):
    #data='kaggle-facemask.yaml'
    #weights='yolov3tiny_facemask.pt'
    #imgsz=640
    with open(data) as f:
        data = yaml.safe_load(f)
    check_dataset(data)
    nc = int(data['nc'])
    model = attempt_load(weights, map_location='cpu')
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(imgsz, s=gs)  # check img_size
    task='val'
    dataloader = create_dataloader(data[task], imgsz, 1, gs, False, pad=0.5, rect=rect,prefix=colorstr(f'{task}: '))[0]
    dataloader_iter=iter(dataloader)
    return model,dataloader_iter


float_model,dataloader_iter=load_model_data('models/yolov3-tiny-facemask.yaml','./models_files/yolov3tiny_facemask.pt',416,False)
print(float_model)
print_size_of_model(float_model)

# num_calibration_batches = 10
quant = torch.quantization.QuantStub()
dequant = torch.quantization.DeQuantStub()
quant_model=nn.Sequential(quant,float_model,dequant)# 在全模型开始和结尾加量化和解量化子模块

quant_model = quant_model.to('cpu')
quant_model.qconfig = torch.quantization.default_qconfig
print(quant_model.qconfig)
torch.quantization.prepare(quant_model, inplace=True)
torch.quantization.convert(quant_model, inplace=True)
print(quant_model)
print("Size of model after quantization")
print_size_of_model(quant_model)

state_dict_t = torch.load('./models_files/facemask/yolov3tiny_facemask_quant.pth')
quant_model.load_state_dict(state_dict_t)
quant_model=quant_model.to('cpu')

# dataset = LoadImages('data/images/bus.jpg', img_size=416, stride=32)
# dataset = LoadImages('data\images\zidane.jpg', img_size=416, stride=32)
dataset = LoadImages('data/images/test.jpg', img_size=416, stride=32)


for path, img, im0s, vid_cap in dataset:
    img = torch.from_numpy(img).to('cpu')
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # img_o = np.array(img)[0,0,:,:]
    # np.savetxt('img_q.txt',img_o)
# data = cv2.imread('data\images\zidane.jpg')

# data,rate,d = letterbox(data,(416, 416))
# # cv2.imshow('input_image', data)
# # cv2.waitKey(0) 
# data = np.expand_dims(data.transpose(2,0,1),axis = 0)
# data = np.expand_dims(data,axis = 0)
# data = torch.from_numpy(data)
quant_model_evaluate_show(img,quant_model)