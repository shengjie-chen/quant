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
from collections import OrderedDict
import math
from scipy import signal
# from yolov3tiny_quant import quant_model_evaluate_show

from quant_utils import load_model_data, quant_model_evaluate_show_name, print_size_of_model, \
    generate_quant_model_baseline,load_float_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='voc_retrain',
                        help='which model? choose in voc_retrain | coco_origin | coco128_retrain ')

    parser.add_argument('--cs_self_def', type=bool, default=False,
                        help='calibration source:a folder including some source imgs to feed quantization model defined manually')
    parser.add_argument('--cs_use_set', type=str, default='val',
                        help='if cs_self_def is false, use dataset as calibration source, you can choose in val and train')
    parser.add_argument('--cs_num', type=int, default=128,
                        help='limit img number of calibration source')

    parser.add_argument('--source', type=str, default='data/images/zidane.jpg',
                        help='a folder of image to detect in quantization model')
    parser.add_argument('--save_qw', action='store_true',
                        help='save quanted weight')

    opt = parser.parse_args()
    print(opt)

    # 加载浮点模型
    model_name = opt.model_name
    data = 'models/yolov3-tiny_voc.yaml' if model_name == 'voc_retrain' else 'models/yolov3-tiny.yaml'
    float_weight = 'models_files/' + model_name + '/weights/best.pt'
    if os.path.exists(float_weight):
        # float_model, dataloader_iter = load_model_data(
        #     data, float_weight, 416, False)
        float_model = load_float_model(float_weight)
    else:
        print("float weight file dont exist!")
    # float_model,dataloader_iter=load_model_data('yolov3-tiny.yaml','model_files/yolov3-tiny.pt',416,False)
    names = float_model.module.names if hasattr(
        float_model, 'module') else float_model.names  # get class names

    # 准备校准数据
    if opt.cs_self_def:
        perpare_source = opt.perpare_source
    else:
        dataset_yaml = 'data/' + model_name.split('_')[0]+'.yaml'
        with open(dataset_yaml) as f:
            data_yaml = yaml.safe_load(f)
        perpare_source = data_yaml[opt.cs_use_set]
    calibrate_dataset = LoadImages(perpare_source, img_size=416, stride=32)

    # # 构建量化模型
    # quant = torch.quantization.QuantStub()
    # dequant = torch.quantization.DeQuantStub()
    # quant_model = nn.Sequential(
    #     quant, float_model, dequant)  # 在全模型开始和结尾加量化和解量化子模块
    # quant_model = quant_model.to('cpu')
    # quant_model.eval()
    # quant_model.qconfig = torch.quantization.default_qconfig
    # print(quant_model.qconfig)
    # model_prepared = torch.quantization.prepare(quant_model)

    # # 喂数据
    # # 对dataset中的图片转为tensor并将范围从0 - 255 to 0.0 - 1.0等等

    # # dataset = LoadImages('data/images', img_size=416, stride=32)
    # # dataset = LoadImages('data/images/v2-90f21023b3bbdb2df60e349d3f6ec279_r.jpg', img_size=416, stride=32)
    # for i, (path, img, im0s, vid_cap) in enumerate(calibrate_dataset):
    #     if i == opt.cs_num:
    #         break
    #     img = torch.from_numpy(img).to('cpu')
    #     img = img.float()  # uint8 to fp16/32
    #     img /= 255.0  # 0 - 255 to 0.0 - 1.0
    #     if img.ndimension() == 3:
    #         img = img.unsqueeze(0)
    #     model_prepared(img)

    # quant_model = torch.quantization.convert(model_prepared)
    quant_model = generate_quant_model_baseline(float_model, calibrate_dataset, opt.cs_num)

    # 保存量化模型
    if opt.save_qw:
        state_dict = quant_model.state_dict()
        torch.save(state_dict, './models_files/' +
                   model_name + '/yolov3tiny_quant.pth')
    # print(quant_model)
    print("Size of model before quantization:")
    print_size_of_model(float_model)
    print("Size of model after quantization:")
    print_size_of_model(quant_model)

    # 使用量化模型检测
    dataset = LoadImages(opt.source, img_size=416, stride=32)
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to('cpu')
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        quant_model_evaluate_show_name(img, quant_model, im0s, names)

    # 使用facemask模型
    # float_model,dataloader_iter=load_model_data('yolov3-tiny-facemask.yaml','models_files/yolov3tiny_facemask.pt',416,False)
    # print(float_model)
    # print("Size of model before quantization:")
    # print_size_of_model(float_model)

    # # dataset = LoadImages('data/images/zidane.jpg', img_size=416, stride=32)
    # # dataset = LoadImages('data/images/bus.jpg', img_size=416, stride=32)
    # # dataset = LoadImages('E:\Academic_study\competition\JiChuang6th\kaggle_facemark\images\maksssksksss1.png', img_size=416, stride=32)
    # dataset = LoadImages('data/images/test_1.jpg', img_size=416, stride=32)

    # for path, img, im0s, vid_cap in dataset:
    #     img = torch.from_numpy(img).to('cpu')
    #     img = img.float()  # uint8 to fp16/32
    #     img /= 255.0  # 0 - 255 to 0.0 - 1.0
    #     if img.ndimension() == 3:
    #         img = img.unsqueeze(0)
    #     # img_o = np.array(img)[0,0,:,:]
    #     # np.savetxt('img_q.txt',img_o)
    # # data = cv2.imread('data\images\zidane.jpg')

    # # num_calibration_batches = 10
    # state_dict_t = torch.load('./models_files/yolov3tiny_facemask_quant.pth')
    # # x = OrderedDict()
    # # for idx, key in enumerate(state_dict_t):
    # #     if 0 <= idx < 2:
    # #         x[key] = state_dict_t[key]

    # quant = torch.quantization.QuantStub()
    # dequant = torch.quantization.DeQuantStub()
    # # quant.load_state_dict(x)
    # quant_model=nn.Sequential(quant,float_model,dequant)# 在全模型开始和结尾加量化和解量化子模块

    # quant_model = quant_model.to('cpu')
    # quant_model.eval()
    # quant_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    # # model_fused = torch.quantization.fuse_modules(quant_model, [['conv', 'bn']])
    # print(quant_model.qconfig)
    # model_prepared = torch.quantization.prepare(quant_model)
    # # torch.quantization.prepare(quant_model, inplace=True)
    # model_prepared(img)
    # quant_model = torch.quantization.convert(model_prepared)
    # quant_model.load_state_dict(state_dict_t)
    # # torch.quantization.convert(quant_model, inplace=True)

    # # print('Post Training Quantization: Convert done')
    # # print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n',quant_model[1].conv)
    # print(quant_model)
    # print("Size of model after quantization:")
    # print_size_of_model(quant_model)

    # # top1, top5 = evaluate(quant_model, criterion, data_loader_test, neval_batches=num_eval_batches)
    # # print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))

    # # data,rate,d = letterbox(data,(416, 416))
    # # # cv2.imshow('input_image', data)
    # # # cv2.waitKey(0)
    # # data = np.expand_dims(data.transpose(2,0,1),axis = 0)
    # # data = np.expand_dims(data,axis = 0)
    # # data = torch.from_numpy(data)
    # quant_model_evaluate_show(img,quant_model)
