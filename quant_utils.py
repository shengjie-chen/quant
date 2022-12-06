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
# from models.yolo import Model
from utils.datasets import create_dataloader, letterbox
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import intersect_dicts, select_device, time_synchronized

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from collections import OrderedDict
import math
from scipy import signal
from models.common import Conv

# from models.quant_yolo import attempt_load_quant

per_channel_qconfig_1 = torch.quantization.QConfig(
    activation=torch.quantization.default_histogram_observer,
    weight=torch.quantization.default_per_channel_weight_observer)
per_channel_qconfig_2 = torch.quantization.QConfig(
    activation=torch.quantization.HistogramObserver.with_args(
        reduce_range=False),
    weight=torch.quantization.default_per_channel_weight_observer)
per_channel_qconfig_3 = torch.quantization.QConfig(
    activation=torch.quantization.default_histogram_observer,
    weight=torch.quantization.PerChannelMinMaxObserver.with_args(
        dtype=torch.qint8, qscheme=torch.per_channel_affine))
per_channel_qconfig_4 = torch.quantization.QConfig(
    activation=torch.quantization.HistogramObserver.with_args(
        quant_min=0, quant_max=191),
    weight=torch.quantization.default_per_channel_weight_observer)
per_channel_qconfig_best = torch.quantization.QConfig(
    activation=torch.quantization.HistogramObserver.with_args(
        quant_min=0, quant_max=255),
    weight=torch.quantization.default_per_channel_weight_observer)
per_channel_qconfig_minmax = torch.quantization.QConfig(
    activation=torch.quantization.MinMaxObserver.with_args()
    ,
    weight=torch.quantization.default_per_channel_weight_observer)


# def generate_quant_model_baseline(model, dataloader_iter, quant_model_name):
#     quant = torch.quantization.QuantStub()
#     dequant = torch.quantization.DeQuantStub()
#     quant_model = nn.Sequential(quant, model, dequant)
#     quant_model = quant_model.to('cpu')
#     quant_model.qconfig = torch.quantization.default_qconfig
#     quant_model = torch.quantization.prepare(quant_model, inplace=True)
#     with torch.no_grad():
#         for t in range(100):
#             print('-', end='')
#             data = next(dataloader_iter)
#             img = data[0]
#             x = img.float()/255.0
#             x = quant_model(x)
#     quant_model = torch.quantization.convert(quant_model, inplace=True)
#     torch.save(quant_model.state_dict(), quant_model_name)

def generate_quant_model_baseline(model, dataloader_iter, cal_num=100):
    """指定量化策略来产生一个量化模型,使用default_qconfig量化策略作为baseline"""
    quant = torch.quantization.QuantStub()
    dequant = torch.quantization.DeQuantStub()
    quant_model = nn.Sequential(quant, model, dequant)
    quant_model = quant_model.to('cpu')
    quant_model.qconfig = torch.quantization.default_qconfig
    print("\t", end='')
    print(quant_model.qconfig)
    quant_model = torch.quantization.prepare(quant_model, inplace=False)
    with torch.no_grad():
        for i, (path, img, im0s, vid_cap) in enumerate(dataloader_iter):
            if i == cal_num-1:
                break
            img = torch.from_numpy(img).to('cpu')
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            quant_model(img)
    quant_model = torch.quantization.convert(quant_model, inplace=False)
    return quant_model


def generate_quant_model_selfdefine(model, dataloader_iter, cal_num=2, act_qmin=0, act_qmax=255):
    """指定量化策略来产生一个量化模型,使用自定义的量化策略"""
    # quant = torch.quantization.QuantStub()
    # dequant = torch.quantization.DeQuantStub()
    # quant_model = nn.Sequential(quant, model, dequant)
    if not hasattr(model, 'quant'):
        model.quant = torch.quantization.QuantStub()
    if not hasattr(model.model[-1], 'dequant'):
        model.model[-1].dequant = torch.quantization.DeQuantStub()
    quant_model = model
    quant_model = quant_model.to('cpu')
    # quant_model.qconfig = torch.quantization.default_per_channel_qconfig
    # quant_model.qconfig = torch.quantization.default_qconfig
    # my_qconfig = torch.quantization.QConfig(activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8),
    #   weight=torch.quantization.default_observer.with_args(dtype=torch.qint8))

    quant_model.qconfig = torch.quantization.QConfig(
        activation=torch.quantization.HistogramObserver.with_args(
            quant_min=act_qmin, quant_max=act_qmax),
        weight=torch.quantization.default_per_channel_weight_observer)
    print("\t", end='')
    print(quant_model.qconfig)
    quant_model = torch.quantization.prepare(quant_model, inplace=False)
    with torch.no_grad():
        for i, (path, img, im0s, vid_cap) in enumerate(dataloader_iter):
            if i == cal_num-1:
                break
            img = torch.from_numpy(img).to('cpu')
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            quant_model(img, quant=2)
    quant_model = torch.quantization.convert(quant_model, inplace=False)
    return quant_model


def generate_quant_model_selfdefine_train(model):
    """指定量化策略来产生一个量化模型,使用自定义的量化策略,用于train脚本"""
    if not hasattr(model, 'quant'):
        model.quant = torch.quantization.QuantStub()
    if not hasattr(model.model[-1], 'dequant'):
        model.model[-1].dequant = torch.quantization.DeQuantStub()
    quant_model = model
    modules_to_fuse = ['conv', 'bn']
    for i, w in enumerate(quant_model.model):
        if isinstance(w, Conv):
            quant_model.model[i] = torch.ao.quantization.fuse_modules_qat(
                quant_model.model[i], modules_to_fuse)
    quant_model = quant_model.to('cpu')
    quant_model.train()
    quant_model.qconfig = per_channel_qconfig_best
    print("\t", end='')
    print(quant_model.qconfig)
    quant_model = torch.quantization.prepare_qat(quant_model, inplace=False)
    return quant_model

def load_quant_model(float_model_name, quant_model_name):
    model = attempt_load(float_model_name, map_location='cpu', inplace=False)
    if not hasattr(model, 'quant'):
        model.quant = torch.quantization.QuantStub()
    if not hasattr(model.model[-1], 'dequant'):
        model.model[-1].dequant = torch.quantization.DeQuantStub()
    quant_model = model

    quant_model = quant_model.to('cpu')
    # quant_model.qconfig = torch.quantization.default_qconfig
    quant_model.qconfig = per_channel_qconfig_best
    quant_model = torch.quantization.prepare(quant_model, inplace=False)
    quant_model = torch.quantization.convert(quant_model, inplace=False)

    state_dict_t = torch.load(quant_model_name)
    quant_model.load_state_dict(state_dict_t)
    quant_model = quant_model.to('cpu')
    return quant_model


def load_float_model(weights):
    """加载浮点模型"""
    model = attempt_load(weights, map_location='cpu', inplace=False)
    return model


def load_model_data(data, weights, imgsz, rect):
    """加载浮点模型以及生成数据加载器"""
    # data='kaggle-facemask.yaml'
    # weights='yolov3tiny_facemask.pt'
    # imgsz=640
    with open(data) as f:
        data = yaml.safe_load(f)
    check_dataset(data)
    nc = int(data['nc'])
    model = attempt_load(weights, map_location='cpu', inplace=False)
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(imgsz, s=gs)  # check img_size
    task = 'val'
    dataloader = create_dataloader(
        data[task], imgsz, 1, gs, False, pad=0.5, rect=rect, prefix=colorstr(f'{task}: '))[0]
    dataloader_iter = iter(dataloader)
    return model, dataloader_iter


def quant_zeropad2d(x):
    """
    实现ZeroPad2d(padding=[0, 1, 0, 1], value=0.0),
    输入的tensor.size()为：[bs,channel,h,w]
    """
    x_scale = x.q_scale()
    x_zq = x.q_zero_point()
    x_type = x.dtype
    x = x.dequantize()
    bs = x.shape[0]
    ch = x.shape[1]
    h = x.shape[2]
    # print(x)
    p = torch.zeros(bs*ch*h)
    p = p.reshape(bs, ch, h, 1)
    x = torch.cat((x, p), 3)
    w = x.shape[3]
    p = torch.zeros(bs*ch*w)
    p = p.reshape(bs, ch, 1, w)
    x = torch.cat((x, p), 2)
    x = torch.quantize_per_tensor(
        x, scale=x_scale, zero_point=x_zq, dtype=x_type)
    return x


def quant_cat(x, y):
    """
    实现cat(),将两个量化的tensor在channel方向上拼接，量化权重使用第一个的
    输入的tensor.size()为：[bs,channel,h,w]
    """
    x_scale = x.q_scale()
    x_zq = x.q_zero_point()
    x_type = x.dtype
    # y_scale = y.q_scale()
    # y_zq = y.q_zero_point()
    # y_type = y.dtype
    x = x.dequantize()
    y = y.dequantize()
    assert (x.shape[0] == y.shape[0])
    assert (x.shape[2] == y.shape[2])
    assert (x.shape[3] == y.shape[3])
    # bs = x.shape[0]
    # ch = x.shape[1]
    # h = x.shape[2]
    # w = x.shape[3]
    x = torch.cat((x, y), 1)
    x = torch.quantize_per_tensor(
        x, scale=x_scale, zero_point=x_zq, dtype=x_type)
    return x


def quant_model_detect_v3t1ancher(x, quant_model):
    """针对yolov3-tiny,使用量化模型进行推理检测,但是只使用一个anchor"""
    x = quant_model[0](x)
    for ii in range(11):
        x = quant_model[1].model[ii](x)
    x = quant_zeropad2d(x)
    for ii in range(12, 16):
        x = quant_model[1].model[ii](x)
    x = quant_model[1].model[-1].m[1](x)
    return x


def quant_model_detect_v3tall(x, quant_model):
    """针对yolov3-tiny,使用量化模型进行推理检测,使用全部anchor"""
    x = quant_model(x, quant=1)
    return x
    # print(quant_model)

    x = quant_model[0](x)
# route 1
    for ii in range(9):
        x = quant_model[1].model[ii](x)
    x2 = x
    for ii in range(9, 11):
        x = quant_model[1].model[ii](x)
    x = quant_zeropad2d(x)
    for ii in range(12, 15):
        x = quant_model[1].model[ii](x)
    x3 = x
    x = quant_model[1].model[15](x)
    x1 = quant_model[1].model[-1].m[1](x)
# route 2
    x3 = quant_model[1].model[16](x3)
    x3 = quant_model[1].model[17](x3)
    # print("before cat:\n\tx3.scale:%f\tx3.zeropoint:%f\n\tx2.scale:%f\tx2.zeropoint:%f"%(x3.q_scale(),x3.q_zero_point(),x2.q_scale(),x2.q_zero_point()))
    x3 = quant_model[1].model[18]([x3, x2])
    # x3 = quant_cat(x3,x2)
    # print("after cat:\n\tx3.scale:%f\tx3.zeropoint:%f"%(x3.q_scale(),x3.q_zero_point()))
    x3 = quant_model[1].model[19](x3)
    x2 = quant_model[1].model[-1].m[0](x3)
    return [x2, x1]

# def quant_inference_batch(x,quant_model):
#     """使用量化模型对一个batchsize的图片进行推理"""
#     bs = x.size()[0]
#     y = []
#     for i in range(bs):
#         m = quant_model_detect_v3t1ancher(x[i], quant_model)
#         y.append(torch.dequantize(m))
#     z = torch.tensor(y)
#     return z


def statistics_per_img(out, targets, paths, seen, stats, niou, img, shapes, device, iouv):
    """对网络结果进行统计"""
    for si, pred in enumerate(out):
        labels = targets[targets[:, 0] == si, 1:]
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
        scale_coords(img[si].shape[1:], predn[:, :4], shapes[si]
                     [0], shapes[si][1])  # native-space pred

        # Assign all predictions as incorrect
        correct = torch.zeros(
            pred.shape[0], niou, dtype=torch.bool, device=device)
        if nl:
            detected = []  # target indices
            tcls_tensor = labels[:, 0]

            # target boxes
            tbox = xywh2xyxy(labels[:, 1:5])
            scale_coords(img[si].shape[1:], tbox, shapes[si]
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
                            correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                            if len(detected) == nl:  # all targets already located in image
                                break

        # Append statistics (correct, conf, pcls, tcls)
        stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
    return stats


def quant_model_evaluate_show_name(data, quant_model, im0s, names):
    """量化模型检测图片并展示类别结果"""
    na = quant_model[1].model[-1].na
    nl = quant_model[1].model[-1].nl
    no = quant_model[1].model[-1].no
    stride = quant_model[1].model[-1].stride
    anchor = quant_model[1].model[-1].anchors
    anchor_grid = quant_model[1].model[-1].anchor_grid

    x = data
    height = im0s.shape[0]
    width = im0s.shape[1]
    while (height > 1000 or width > 1500):
        height = int(height / 2)
        width = int(width / 2)
    res = quant_model_detect_v3t1ancher(x, quant_model)
    pred_reduce = torch.dequantize(res)

    bs, _, ny, nx = pred_reduce.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
    y = pred_reduce.view(bs, na, no, ny, nx).permute(
        0, 1, 3, 4, 2).contiguous()

    y = y.sigmoid()
    grid = quant_model[1].model[-1]._make_grid(nx, ny)
    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid) * stride[1]  # xy
    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[1]  # wh
    y = y.view(1, -1, no)
    # print(y)
    y_nms = non_max_suppression(y, 0.25, 0.45, None, False, max_det=1000)

    for i, det in enumerate(y_nms):  # detections per image
        im0 = im0s.copy()
        imc = im0
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(
                data.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                # Add bbox to image
                c = int(cls)  # integer class
                label = f'{names[c]} {conf:.2f}'
                plot_one_box(xyxy, im0, label=label,
                             color=colors(c, True), line_thickness=3)

    cv2.namedWindow('input_image', 0)
    cv2.resizeWindow('input_image', width, height)
    cv2.imshow('input_image', im0)
    cv2.waitKey(0)


def quant_model_evaluate_show(data, quant_model):
    na = quant_model[1].model[-1].na
    nl = quant_model[1].model[-1].nl
    no = quant_model[1].model[-1].no
    stride = quant_model[1].model[-1].stride
    anchor = quant_model[1].model[-1].anchors
    anchor_grid = quant_model[1].model[-1].anchor_grid

    # img=data[0]
    # x=(img.float()/255.0)
    x = data

    res = quant_model_detect_v3t1ancher(x, quant_model)
    pred_reduce = torch.dequantize(res)

    bs, _, ny, nx = pred_reduce.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
    y = pred_reduce.view(bs, na, no, ny, nx).permute(
        0, 1, 3, 4, 2).contiguous()

    y = y.sigmoid()
    grid = quant_model[1].model[-1]._make_grid(nx, ny)
    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid) * stride[1]  # xy
    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[1]  # wh
    y = y.view(1, -1, no)
    print(y)
    y_nms = non_max_suppression(y, 0.25, 0.5, None, False, max_det=1000)

    img_numpy = data[0, :, :, :].numpy()
    img_numpy = img_numpy.swapaxes(0, 1)
    img_numpy = img_numpy.swapaxes(1, 2)
    img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
    cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE)
    for ii in y_nms[0][:, :4].round():
        x1, y1, x2, y2 = ii[0].int().item(), ii[1].int(
        ).item(), ii[2].int().item(), ii[3].int().item()
        cv2.rectangle(img_numpy, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow('input_image', img_numpy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def generate_para_list(quant_model, index):
    cscale = quant_model.state_dict(
    )['1.model.'+str(index)+'.conv.scale'].item()
    ascale = quant_model.state_dict(
    )['1.model.'+str(index)+'.act.scale'].item()
    czp = quant_model.state_dict(
    )['1.model.'+str(index)+'.conv.zero_point'].item()
    azp = quant_model.state_dict(
    )['1.model.'+str(index)+'.act.zero_point'].item()
    wscale = torch.q_scale(quant_model.state_dict()[
                           '1.model.'+str(index)+'.conv.weight'])
    w = quant_model.state_dict(
    )['1.model.'+str(index)+'.conv.weight'].int_repr().numpy().astype(np.int8)
    b = quant_model.state_dict()['1.model.'+str(index)+'.conv.bias']
    return w, b, wscale, ascale, cscale, azp, czp


def generate_quant_para(iscale, izp, wscale, ascale, cscale, azp, czp):
    # 缩写的含义分别为input scale,input zero_point,weight scale,activation scale,conv scale,activation zero_point,conv zero_point
    oscale = cscale
    ozp = czp
    bscale = iscale * wscale
    m = iscale*wscale/oscale
    base, expr = math.frexp(m)
    mult = round(base*(2**15))
    shift = -expr
    return oscale, ozp, bscale, mult, shift


def generate_weight(weight):
    ifmch = weight.shape[1]
    weight = weight.reshape([-1, 8, ifmch, 9])
    if ifmch < 8:
        tmp = np.zeros([2, 8, 5, 9]).astype(np.int8)
        weight = np.append(weight, tmp, axis=2)
    weight = weight.swapaxes(1, 2)
    weight = weight.swapaxes(2, 3)
    return weight


def generate_weight_1x1(weight):
    ifmch = weight.shape[1]
    ofmch = weight.shape[0]
    weight = weight.reshape([ofmch, ifmch])
    weight_zero = np.zeros([ofmch, ifmch, 9]).astype(np.int8)
    weight_zero[:, :, 4] = weight
    weight = weight_zero
    weight = weight.reshape([-1, 8, ifmch, 9])
    weight = weight.swapaxes(1, 2)
    weight = weight.swapaxes(2, 3)
    return weight


def generate_weight_detect(weight, ofmch_t):
    ifmch = weight.shape[1]
    ofmch = weight.shape[0]

    weight = weight.reshape([ofmch, ifmch])
    weight_zero = np.zeros([ofmch, ifmch, 9]).astype(np.int8)
    weight_zero[:, :, 4] = weight
    weight = weight_zero

    # print(weight.shape)
    ofmch = weight.shape[0]
    concat = np.zeros([ofmch_t-ofmch, ifmch, 9]).astype(np.int8)
    # print(concat.shape)
    weight = np.concatenate((weight, concat), axis=0)
    # print(weight.shape)

    weight = weight.reshape([-1, 8, ifmch, 9])
    weight = weight.swapaxes(1, 2)
    weight = weight.swapaxes(2, 3)

    # print(weight.shape)
    return weight

# 每个候选框的深度要保证8的倍数，85的话去掉5或者增加3个通道


def generate_weight_detect_8x(weight, ofmch_t):
    ifmch = weight.shape[1]
    ofmch = weight.shape[0]

    weight = weight.reshape([ofmch, ifmch])
    zero = np.zeros([3, ifmch]).astype(np.int8)
    weight = np.concatenate(
        (weight[0:85, :], zero, weight[85:170, :], zero, weight[170:256, :], zero), axis=0)
    ofmch = weight.shape[0]
    weight_zero = np.zeros([ofmch, ifmch, 9]).astype(np.int8)
    weight_zero[:, :, 4] = weight
    weight = weight_zero

    # print(weight.shape)
    ofmch = weight.shape[0]
    concat = np.zeros([ofmch_t-ofmch, ifmch, 9]).astype(np.int8)
    # print(concat.shape)
    weight = np.concatenate((weight, concat), axis=0)
    # print(weight.shape)

    weight = weight.reshape([-1, 8, ifmch, 9])
    weight = weight.swapaxes(1, 2)
    weight = weight.swapaxes(2, 3)

    # print(weight.shape)
    return weight


def generate_weight_detect_75to72(weight, ofmch_t):
    """对于voc模型最后一层，将75个滤波器权重减少为72个"""
    index_w = [24, 49, 74]
    weight = np.delete(weight, index_w, axis=0)
    ifmch = weight.shape[1]
    ofmch = weight.shape[0]

    weight = weight.reshape([ofmch, ifmch])

    # zero=np.zeros([3,ifmch]).astype(np.int8)
    # weight=np.concatenate((weight[0:85,:],zero,weight[85:170,:],zero,weight[170:256,:],zero),axis=0)
    ofmch = weight.shape[0]
    weight_zero = np.zeros([ofmch, ifmch, 9]).astype(np.int8)
    weight_zero[:, :, 4] = weight
    weight = weight_zero

    # print(weight.shape)
    ofmch = weight.shape[0]
    concat = np.zeros([ofmch_t-ofmch, ifmch, 9]).astype(np.int8)
    # print(concat.shape)
    weight = np.concatenate((weight, concat), axis=0)
    # print(weight.shape)

    weight = weight.reshape([-1, 8, ifmch, 9])
    weight = weight.swapaxes(1, 2)
    weight = weight.swapaxes(2, 3)

    # print(weight.shape)
    return weight


def generate_bias(b, bscale):
    bias = torch.quantize_per_tensor(
        b, scale=bscale, zero_point=0, dtype=torch.qint32).int_repr().numpy().astype(np.int64)
    return bias


def generate_leakytbl(act_scale, act_zp, output_scale, output_zp):
    sm = output_scale/act_scale
    lista = np.array(range(256))
    table = []
    for data in lista:
        if data < output_zp:
            tmp = round((data-output_zp)*sm*0.1+act_zp)
        else:
            tmp = round((data-output_zp)*sm+act_zp)
        if tmp >= 255:
            tmp = 255
        table.append(tmp)
    table = np.array(table).astype(np.uint64)
    return table


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_sigmoid_table(zp, scale, rp):
    sigmoid_tbl = sigmoid((np.array(range(256))-zp)*scale)
    np.savetxt(rp+'/sigmoid_table.h', sigmoid_tbl,
               delimiter=',', encoding='utf-8')
    return sigmoid_tbl


def yolov3tiny_infer_para_gen(quant_model, ofmch_t, first_chr, rp, model_name):
    convindex = [0, 2, 4, 6, 8, 10, 13, 14, 15]
    convtype = [0, 0, 0, 0, 0, 0, 0, 1, 0]
    mult_list = []
    shift_list = []
    izp_list = []
    ozp_list = []
    azp_list = []

    ascale = quant_model.state_dict()['0.scale'].item()
    azp = quant_model.state_dict()['0.zero_point'].item()
    cnt = 0

    for index in convindex:
        iscale = ascale
        izp = azp
        w, b, wscale, ascale, cscale, azp, czp = generate_para_list(
            quant_model, index)
        oscale, ozp, bscale, mult, shift = generate_quant_para(
            iscale, izp, wscale, ascale, cscale, azp, czp)
        if convtype[cnt] == 1:
            weight = generate_weight_1x1(w)
        else:
            weight = generate_weight(w)
        bias = generate_bias(b, bscale)
        leakytbl = generate_leakytbl(ascale, azp, oscale, ozp)
        mult_list.append(mult)
        shift_list.append(shift)
        izp_list.append(izp)
        ozp_list.append(ozp)
        azp_list.append(azp)
        weight.tofile(rp + '/infer_bin/'+first_chr+'W'+str(cnt)+'.bin')
        bias.tofile(rp + '/infer_bin/'+first_chr+'B'+str(cnt)+'.bin')
        leakytbl.tofile(rp + '/infer_bin/'+first_chr+'R'+str(cnt)+'.bin')
        # print(cnt,index,w.shape,weight.shape)
        cnt = cnt+1

    iscale = ascale
    izp = azp
    oscale = quant_model.state_dict()['1.model.20.m.1.scale'].item()
    ozp = quant_model.state_dict()['1.model.20.m.1.zero_point'].item()
    wscale = torch.q_scale(quant_model.state_dict()['1.model.20.m.1.weight'])
    m = iscale*wscale/oscale
    base, expr = math.frexp(m)
    mult = round(base*(2**15))
    shift = -expr

    w = quant_model.state_dict(
    )['1.model.20.m.1.weight'].int_repr().numpy().astype(np.int8)
    b = quant_model.state_dict()['1.model.20.m.1.bias']
    bscale = iscale * wscale
    bias = torch.quantize_per_tensor(
        b, scale=bscale, zero_point=0, dtype=torch.qint32).int_repr().numpy().astype(np.int64)
    # zero=np.zeros([3]).astype(np.int64)
    # bias=np.concatenate((bias[0:85],zero,bias[85:170],zero,bias[170:256],zero))
    if (model_name == 'voc_retrain'):
        index_bias = [24, 49, 74]
        bias = np.delete(bias, index_bias)
        index_w = [24, 49, 74]
        w = np.delete(w, index_w, axis=0)

    weight = generate_weight_detect(w, ofmch_t)
    # if(first_chr=='Y'):
    #     weight=generate_weight_detect_8x(w,ofmch_t)
    # else:
    #     weight=generate_weight_detect(w,ofmch_t)
    # weight=generate_weight_detect(w,ofmch_t)
    leakytbl = np.array(range(256)).astype(np.uint64)
    sigmoid_scale = quant_model.state_dict()['1.model.20.m.1.scale'].item()
    sigmoid_zp = quant_model.state_dict()['1.model.20.m.1.zero_point'].item()
    print("sigmoid_scale=%.20f" % sigmoid_scale)
    print("sigmoid_zp=%d" % sigmoid_zp)
    sigmoid_tbl = generate_sigmoid_table(sigmoid_zp, sigmoid_scale, rp)

    mult_list.append(mult)
    shift_list.append(shift)
    izp_list.append(izp)
    ozp_list.append(ozp)
    azp_list.append(ozp)

    weight.tofile(rp + '/infer_bin/'+first_chr+'W'+str(cnt)+'.bin')
    bias.tofile(rp + '/infer_bin/'+first_chr+'B'+str(cnt)+'.bin')
    leakytbl.tofile(rp + '/infer_bin/'+first_chr+'R'+str(cnt)+'.bin')
    sigmoid_tbl.astype(np.float32).tofile(
        rp + '/infer_bin/'+first_chr+'SG.bin')

    sel_in_list = [5, 4, 3, 2, 1, 0, 0, 0, 0, 0]
    pool_list = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    stride_list = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    ifm_list = [8, 16, 32, 64, 128, 256, 512, 1024, 256, 512]

    ofm_list = [16, 32, 64, 128, 256, 512, 1024, 256, 512, ofmch_t]
    config_list = ifm_list+ofm_list+mult_list+shift_list + \
        izp_list+ozp_list+azp_list+sel_in_list+pool_list+stride_list
    config_list = np.array(config_list).astype(np.uint32).flatten()
    config_list.astype(np.uint32).tofile(
        rp + '/infer_bin/'+first_chr+'CG'+'.bin')

    print('mult:', mult_list)
    print('shift:', shift_list)
    print('zpin:', izp_list)
    print('zpout:', ozp_list)
    print('zpact:', azp_list)
    # print('list:',config_list)
    return config_list


def model_quant_save(dataset, float_model, quant_model_files):
    quant = torch.quantization.QuantStub()
    dequant = torch.quantization.DeQuantStub()
    quant_model = nn.Sequential(
        quant, float_model, dequant)  # 在全模型开始和结尾加量化和解量化子模块
    quant_model = quant_model.to('cpu')
    quant_model.eval()
    quant_model.qconfig = torch.quantization.default_qconfig
    print(quant_model.qconfig)
    model_prepared = torch.quantization.prepare(quant_model)

    # 对dataset中的图片转为tensor并将范围从0 - 255 to 0.0 - 1.0等等
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
        model_prepared(img)

    quant_model = torch.quantization.convert(model_prepared)
    state_dict = quant_model.state_dict()
    torch.save(state_dict, quant_model_files)
    print(quant_model)
    print("Size of model before quantization:")
    print_size_of_model(float_model)
    print("Size of model after quantization:")
    print_size_of_model(quant_model)


def _make_grid(nx=20, ny=20):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


def yolo_func(yolo_in, nc, anchors, stride):
    no = nc + 5  # number of outputs per anchor
    nl = len(anchors)  # number of detection layers
    na = len(anchors[0]) // 2  # number of anchors
    grid = [torch.zeros(1)] * nl  # init grid
    a = torch.tensor(anchors).float().view(nl, -1, 2)
    anchor_grid = a.clone().view(nl, 1, -1, 1, 1, 2).to(yolo_in[0].device)
    z = []  # inference output
    yolo_in = list(yolo_in)
    for i in range(nl):
        bs, _, ny, nx = yolo_in[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        yolo_in[i] = yolo_in[i].view(bs, na, no, ny, nx).permute(
            0, 1, 3, 4, 2).contiguous()
        if grid[i].shape[2:4] != yolo_in[i].shape[2:4]:
            grid[i] = _make_grid(nx, ny).to(yolo_in[i].device)
        y = yolo_in[i].sigmoid().to(yolo_in[i].device)
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid[i]) * stride[i]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
        # xy = (y[..., 0:2] * 2. - 0.5 + grid[i]) * stride[i]  # xy
        # wh = (y[..., 2:4] * 2) ** 2 * anchor_grid[i].view(1, na, 1, 1, 2)  # wh
        # y = torch.cat((xy, wh, y[..., 4:]), -1)
        z.append(y.view(bs, -1, no))
    return (torch.cat(z, 1), yolo_in)
