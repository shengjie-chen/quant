"""
批量测试几个重要量化参数对于量化精度的影响
相当于quant_test.py脚本加以下参数下
                "--recon_qmodel",
                "--cs_self_def",
                "--quant_strategy", "selfdefine",
测试主要围绕这四个参数
                "--cs_dir","",
                "--cs_num", "",
                "--act_qmin", "",
                "--act_qmax", "",

"""
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

from quant_test import quant_test
from utils.datasets import create_dataloader
from utils.general import check_dataset,  check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from utils.datasets import LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression,  \
    scale_coords, xyxy2xywh,  set_logging, increment_path
from scipy import signal
# from yolov3tiny_quant import quant_model_evaluate_show
from quant_utils import load_model_data, quant_model_evaluate_show_name, print_size_of_model,\
    load_quant_model, statistics_per_img, load_float_model, quant_model_detect_v3t1ancher, _make_grid, \
    generate_quant_model_baseline, generate_quant_model_selfdefine, quant_model_detect_v3tall
import warnings

from utils.metrics import fitness
import matplotlib.pyplot as plt


def getAccLineChart(x, y, save_dir, chart_name, x_label, y_label):
    plt.cla()
    l = plt.plot(x, y, "b--*")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.legend()
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, chart_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--model_name', type=str, default='voc_retrain',
                        help='which model? choose in voc_retrain | coco_origin | coco128_retrain ')
    parser.add_argument('--data', type=str,
                        default='data/voc.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='size of each image batch')

    parser.add_argument('--task', type=str,
                        default='test', help='train, val or test')

    parser.add_argument('--one_anchor', action='store_true',
                        help='default means use all two anchor, and add the flag means use one anchor.')

    parser.add_argument('--recon_qmodel', action='store_true',
                        help='default means load quant model,and add the flag means reconstruct quant model.')
    parser.add_argument('--cs_self_def', action='store_true',
                        help='calibration source:a folder including some source imgs to feed quantization model defined manually')
    parser.add_argument('--cs_dir', type=str, default='E:/Academic_study/datasets/coco128/images/train2017',
                        help='if cs_self_def is true, use cs_dir as calibration source')
    parser.add_argument('--cs_use_set', type=str, default='val',
                        help='if cs_self_def is false, use dataset as calibration source, you can choose in val and train')
    parser.add_argument('--cs_num', type=int, default=128,
                        help='limit img number of calibration source')

    parser.add_argument('--quant_strategy', type=str, default='baseline',
                        help='choose quant strategy in baseline/selfdefine...')
    parser.add_argument('--save_qw', action='store_true',
                        help='save quanted weight')

    parser.add_argument('--quant_weight', type=str, default=None,
                        help='choose one quant weight, and recon_qmodel must be false')
    parser.add_argument('--float_weight', type=str, default=None,
                        help='choose one float weight, and recon_qmodel must be true')

    parser.add_argument('--act_qmin', type=int, default=0,
                        help='if quant_strategy is selfdefine, use to change quant config activation quant_min')
    parser.add_argument('--act_qmax', type=int, default=255,
                        help='if quant_strategy is selfdefine, use to change quant config activation quant_max')
    opt = parser.parse_args()
    print(opt, type(opt))
    warnings.filterwarnings('ignore')

    # Configure
    device = 'cpu'
    data_yaml = opt.data
    if isinstance(data_yaml, str):
        with open(data_yaml) as f:
            data_cfg = yaml.safe_load(f)
    check_dataset(data_cfg)  # check
    nc = int(data_cfg['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    imgsz = 416
    gs = 32
    names = {k: v for k, v in enumerate(range(nc))}
    anchors = [[10, 14, 23, 27, 37, 58], [81, 82, 135, 169, 344, 319]]
    anchors = [anchors[1], anchors[0]]
    # stride = [32, 16]
    conf_thres = 0.001
    iou_thres = 0.6  # for NMS
    batch_size = opt.batch_size
    model_name = opt.model_name

    data = 'models/yolov3-tiny_voc.yaml' if model_name == 'voc_retrain' else 'models/yolov3-tiny.yaml'
    relative_path = './models_files/' + model_name
    float_weight = relative_path + \
        '/weights/best.pt' if opt.float_weight is None else opt.float_weight
    quant_weight = relative_path + \
        '/yolov3tiny_quant.pth' if opt.quant_weight is None else opt.quant_weight

    assert (os.path.exists(float_weight))
    assert (os.path.exists(quant_weight))
    print("->使用现有浮点权重：\n\t"+str(float_weight))

    # opt test
    # cs_dir_list = ["/SSD/csj/CNN/datasets/VOC/images/test2007",
    #                "/SSD/csj/CNN/datasets/VOC/images/trainval",
    #                "/SSD/csj/CNN/datasets/VOC/images/val",
    #                "/SSD/csj/CNN/coco128/images/train2017"]
    cs_dir_list = ["/SSD/csj/CNN/datasets/VOC/images/val"]
    cs_num_list = np.array([4, 8, 16, 32, 64, 128, 256])
    cs_num_list = np.array([2, 3, 5, 6, 10, 12, 24, 40])

    act_qmin_list = np.array([0, 4, 16, 32])
    act_qmax_list = 128 + np.array([0, 4, 16, 32, 64, 96])

    map50_res = np.empty((len(cs_dir_list), len(cs_num_list)))
    map_res = np.empty((len(cs_dir_list), len(cs_num_list)))
    fitness_res = np.empty((len(cs_dir_list), len(cs_num_list)))
    store_path = './batchtest/test1/'
    store_path = str(increment_path(Path('./batchtest/test')))
    if os.path.exists(store_path) == False:
        os.makedirs(store_path)
    res_file = open(os.path.join(store_path, 'quant_test_res.txt'), 'w')

    for i, dir in enumerate(cs_dir_list):
        dir_suffix = dir.split('/')[-3] + '_' + dir.split('/')[-1]
        for j, cs_num in enumerate(cs_num_list):
            opt.cs_dir = dir
            opt.cs_num = cs_num
            if (opt.cs_dir == "../../datasets/coco128/images/train2017" and opt.cs_num > 128):
                continue
            if (opt.recon_qmodel):  # 重构量化模型
                # 准备校准数据
                if opt.cs_self_def:  # 自定义校准数据
                    perpare_source = opt.cs_dir
                else:
                    dataset_yaml = 'data/' + model_name.split('_')[0]+'.yaml'
                    with open(dataset_yaml) as f:
                        data_yaml = yaml.safe_load(f)
                    perpare_source = data_yaml[opt.cs_use_set]

                print("->重构量化权重：\n\t使用校准数据："+str(perpare_source) +
                      "\n\t图片数量："+str(opt.cs_num))
                calibrate_dataset = LoadImages(
                    perpare_source, img_size=416, stride=32)
                float_model = load_float_model(float_weight)

                if (opt.quant_strategy == 'baseline'):  # 使用默认的量化策略
                    print("->baseline量化策略：")
                    quant_model = generate_quant_model_baseline(
                        float_model, calibrate_dataset, opt.cs_num)
                else:  # 使用自定义的量化策略，产生了量化模型
                    print("->selfdefine量化策略：")
                    quant_model = generate_quant_model_selfdefine(
                        float_model, calibrate_dataset, opt.cs_num, opt.act_qmin, opt.act_qmax)
            else:  # 加载保存好的量化模型
                # load model only in default_qconfig strategy
                quant_model = load_quant_model(float_weight, quant_weight)
                print("加载现有量化权重："+str(quant_weight))

                # 保存量化模型
            if opt.save_qw:
                state_dict = quant_model.state_dict()
                torch.save(state_dict, './models_files/' +
                           model_name + '/yolov3tiny_quant.pth')

            # Configure
            if (opt.quant_strategy == 'baseline'):  # 使用默认的量化策略
                na = quant_model[1].model[-1].na
                # number of detection layers
                nd = 1 if opt.one_anchor else quant_model[1].model[-1].nl
                no = quant_model[1].model[-1].no
                stride = quant_model[1].model[-1].stride
                anchor = quant_model[1].model[-1].anchors
                anchor_grid = quant_model[1].model[-1].anchor_grid
            else:
                na = quant_model.model[-1].na
                # number of detection layers
                nd = 1 if opt.one_anchor else quant_model.model[-1].nl
                no = quant_model.model[-1].no
                stride = quant_model.model[-1].stride
                anchor = quant_model.model[-1].anchors
                anchor_grid = quant_model.model[-1].anchor_grid

            task = opt.task
            dataloader = create_dataloader(data_cfg[task], imgsz, batch_size, gs, opt, pad=0.5, rect=True,
                                           prefix=colorstr(f'{task}: '))[0]

            res = quant_test(quant_model, dataloader, nc, iouv, niou, names,
                             conf_thres, iou_thres, device, opt)
            res = np.array(res).reshape(1, -1)
            map50_res[i, j] = res[0][2]
            map_res[i, j] = res[0][3]
            fitness_res[i, j] = fitness(res)
            res_file.write('cs_dir:%s\t\tcs_num:%d\t->\tmap50:%f\t\tmap:%f\t\tfitness:%f\n' % (
                dir_suffix, cs_num, map50_res[i, j], map_res[i, j], fitness_res[i, j]))

        getAccLineChart(
            cs_num_list, map50_res[i], store_path, dir_suffix+'_map50_res', 'cs_num', 'map50_res')
        getAccLineChart(
            cs_num_list, map_res[i], store_path, dir_suffix+'_map_res', 'cs_num', 'map_res')
        getAccLineChart(cs_num_list, fitness_res[i], store_path,
                        dir_suffix+'_fitness_res', 'cs_num', 'fitness_res')
    np.savetxt(os.path.join(store_path, 'map50_res.txt'), map50_res)
    np.savetxt(os.path.join(store_path, 'map_res.txt'), map_res)
    np.savetxt(os.path.join(store_path, 'fitness_res.txt'), fitness_res)

    res_file.write("\nq_min = %d \nq_max = %d" % (opt.act_qmin, opt.act_qmax))
    res_file.close()
