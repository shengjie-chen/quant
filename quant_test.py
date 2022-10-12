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
from quant_utils import load_model_data, quant_model_evaluate_show_name, print_size_of_model,\
    load_quant_model, statistics_per_img, load_float_model, quant_model_detect, _make_grid
import warnings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--model_name', type=str, default='voc_retrain',
                        help='which model? choose in voc_retrain | coco_origin | coco128_retrain ')
    parser.add_argument('--data', type=str,
                        default='data/voc.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='size of each image batch')

    # parser.add_argument('--weights', nargs='+', type=str,
    #                     default='yolov3.pt', help='model.pt path(s)')

    opt = parser.parse_args()
    # opt.save_json |= opt.data.endswith('coco.yaml')
    # opt.data = check_file(opt.data)  # check file
    print(opt, type(opt))
    warnings.filterwarnings('ignore')

    model_name = opt.model_name
    data = 'models/yolov3-tiny_voc.yaml' if model_name == 'voc_retrain' else 'models/yolov3-tiny.yaml'
    relative_path = './models_files/' + model_name
    float_weight = relative_path + '/weights/best.pt'
    quant_weight = relative_path + '/yolov3tiny_quant.pth'

    # load model
    assert(os.path.exists(float_weight))
    assert(os.path.exists(quant_weight))
    quant_model = load_quant_model(float_weight, quant_weight)

    # Configure
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
    device = 'cpu'
    na = quant_model[1].model[-1].na
    nl = quant_model[1].model[-1].nl
    no = quant_model[1].model[-1].no
    stride = quant_model[1].model[-1].stride
    anchor = quant_model[1].model[-1].anchors
    anchor_grid = quant_model[1].model[-1].anchor_grid

    task = 'val'
    dataloader = create_dataloader(data_cfg[task], imgsz, batch_size, gs, opt, pad=0.5, rect=True,
                                   prefix=colorstr(f'{task}: '))[0]
    best_acc, old_file = 0, None
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images',
                                 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    t_begin = time.time()
    test_loss = torch.zeros(3, device=device)
    stats, ap, ap_class = [], [], []
    correct = 0
    trained_with_quantization = True
    seen = 0

    time1 = time.time()
    # for data, target in test_loader:
    for i, (data, target, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        # print('Now calculating the {}th data...'.format(i+1))
        # print(len(test_loader.dataset[0][0][0]))
        data = data.float()  # uint8 to fp16/32
        data /= 255.0  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = data.shape  # batch size, channels, height, width
        with torch.no_grad():
            # pred_reduce = quant_inference_batch(data, quant_model)
            res = quant_model_detect(data, quant_model)
            pred_reduce = torch.dequantize(res)

            # x(bs,255,20,20) to x(bs,3,20,20,85)
            bs, _, ny, nx = pred_reduce.shape
            y = pred_reduce.view(bs, na, no, ny, nx).permute(
                0, 1, 3, 4, 2).contiguous()

            y = y.sigmoid()
            grid = quant_model[1].model[-1]._make_grid(nx, ny)
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid) * stride[1]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[1]  # wh
            y = y.view(bs, -1, no)
            # print(y)

            target[:, 2:] *= torch.Tensor([width,height, width, height])
            lb = []  # for autolabelling
            y_nms = non_max_suppression(
                y,conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=False, max_det=1000)

            # Statistics per image
            # stats=statistics_per_img(out=y_nms, targets=target, paths=paths,
            #                            seen=seen, stats=stats, niou=niou, img=data, shapes=shapes, device=device, iouv=iouv)
            # 对网络结果进行统计
            for si, pred in enumerate(y_nms):
                labels = target[target[:, 0] == si, 1:]
                nl = len(labels)        # standard obj num in current img
                tcls = labels[:, 0].tolist() if nl else []  # target class
                path = Path(paths[si])
                seen += 1

                if len(pred) == 0:      # pred obj num in current img
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool),
                                      torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # Predictions
                predn = pred.clone()
                scale_coords(data[si].shape[1:], predn[:, :4], shapes[si]
                             [0], shapes[si][1])  # native-space pred

                # Assign all predictions as incorrect
                correct = torch.zeros(
                    pred.shape[0], niou, dtype=torch.bool, device=device)
                if nl:
                    detected = []  # target indices
                    tcls_tensor = labels[:, 0]

                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5])
                    scale_coords(data[si].shape[1:], tbox, shapes[si]
                                 [0], shapes[si][1])  # native-space labels

                    # Per target class
                    for cls in torch.unique(tcls_tensor):
                        ti = (cls == tcls_tensor).nonzero(
                            as_tuple=False).view(-1)  # target indices
                        pi = (cls == pred[:, 5]).nonzero(
                            as_tuple=False).view(-1)  # prediction indices

                        # Search for detections
                        if pi.shape[0]:
                            # Prediction to target ious
                            ious, i = box_iou(predn[pi, :4], tbox[ti]).max(
                                1)  # best ious, indices

                            # Append detections
                            detected_set = set()
                            for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                d = ti[i[j]]  # detected target
                                if d.item() not in detected_set:
                                    detected_set.add(d.item())
                                    detected.append(d)
                                    # iou_thres is 1xn
                                    correct[pi[j]] = ious[j] > iouv
                                    if len(detected) == nl:  # all targets already located in image
                                        break

                # Append statistics (correct, conf, pcls, tcls)
                stats.append(
                    (correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(
            *stats, plot=False, save_dir='.', names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        # number of targets per class
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
