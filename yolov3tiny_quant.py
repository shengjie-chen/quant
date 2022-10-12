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

def generate_quant_model(model,dataloader_iter,quant_model_name):
    quant = torch.quantization.QuantStub()
    dequant = torch.quantization.DeQuantStub()
    quant_model=nn.Sequential(quant,model,dequant)
    quant_model = quant_model.to('cpu')
    quant_model.qconfig = torch.quantization.default_qconfig
    quant_model=torch.quantization.prepare(quant_model, inplace=True)
    with torch.no_grad():
        for t in range(100):
            print('-',end='')
            data=next(dataloader_iter)
            img=data[0]
            x=img.float()/255.0
            x = quant_model(x)
    quant_model=torch.quantization.convert(quant_model, inplace=True)
    torch.save(quant_model.state_dict(),quant_model_name)

def load_quant_model(float_model_name,quant_model_name):
    model = attempt_load(float_model_name, map_location='cpu')
    quant = torch.quantization.QuantStub()
    dequant = torch.quantization.DeQuantStub()
    quant_model=nn.Sequential(quant,model,dequant)

    quant_model = quant_model.to('cpu')
    quant_model.qconfig = torch.quantization.default_qconfig
    quant_model=torch.quantization.prepare(quant_model, inplace=False)
    quant_model=torch.quantization.convert(quant_model, inplace=False)

    state_dict_t = torch.load(quant_model_name)
    quant_model.load_state_dict(state_dict_t)
    quant_model=quant_model.to('cpu')
    return quant_model

def quant_model_detect(x,quant_model):
    x=quant_model[0](x)
    for ii in range(11):
        print(ii,quant_model[1].model[ii])
        x=quant_model[1].model[ii](x)
    for ii in range(12,16):
        print(ii,quant_model[1].model[ii])
        x=quant_model[1].model[ii](x)
    x=quant_model[1].model[-1].m[1](x)
    return x

def quant_model_evaluate_show(data,quant_model):
    na=quant_model[1].model[-1].na
    nl=quant_model[1].model[-1].nl
    no=quant_model[1].model[-1].no
    stride=quant_model[1].model[-1].stride
    anchor=quant_model[1].model[-1].anchors
    anchor_grid=quant_model[1].model[-1].anchor_grid

    img=data[0]
    x=(img.float()/255.0)
    res=quant_model_detect(x,quant_model)
    pred_reduce=torch.dequantize(res)
    
    bs, _, ny, nx = pred_reduce.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
    y = pred_reduce.view(bs, na, no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

    y=y.sigmoid()
    grid=quant_model[1].model[-1]._make_grid(nx, ny)
    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid) * stride[1]  # xy
    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[1]  # wh
    y=y.view(1, -1, no)

    y_nms=non_max_suppression(y, 0.25, 0.5, None, False,max_det=1000)

    img_numpy=img[0,:,:,:].numpy()
    img_numpy=img_numpy.swapaxes(0,1)
    img_numpy=img_numpy.swapaxes(1,2)
    # img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
    cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE)
    for ii in y_nms[0][:,:4].round():
        x1,y1,x2,y2=ii[0].int().item(),ii[1].int().item(),ii[2].int().item(),ii[3].int().item()
        cv2.rectangle(img_numpy, (x1,y1), (x2,y2), (0, 0, 255), 2)
    cv2.imshow('input_image', img_numpy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def generate_para_list(quant_model,index):
    cscale=quant_model.state_dict()['1.model.'+str(index)+'.conv.scale'].item()
    ascale=quant_model.state_dict()['1.model.'+str(index)+'.act.scale'].item()
    czp=quant_model.state_dict()['1.model.'+str(index)+'.conv.zero_point'].item()
    azp=quant_model.state_dict()['1.model.'+str(index)+'.act.zero_point'].item()
    wscale=torch.q_scale(quant_model.state_dict()['1.model.'+str(index)+'.conv.weight'])
    w=quant_model.state_dict()['1.model.'+str(index)+'.conv.weight'].int_repr().numpy().astype(np.int8)
    b=quant_model.state_dict()['1.model.'+str(index)+'.conv.bias']
    return w,b,wscale,ascale,cscale,azp,czp

def generate_quant_para(iscale,izp,wscale,ascale,cscale,azp,czp):
    #缩写的含义分别为input scale,input zero_point,weight scale,activation scale,conv scale,activation zero_point,conv zero_point
    oscale=cscale
    ozp=czp
    bscale = iscale * wscale
    m=iscale*wscale/oscale
    base,expr=math.frexp(m)
    mult=round(base*(2**15))
    shift=-expr
    return oscale,ozp,bscale,mult,shift

def generate_weight(weight):
    ifmch=weight.shape[1]
    weight=weight.reshape([-1,8,ifmch,9])
    if ifmch<8:
        tmp=np.zeros([2,8,5,9]).astype(np.int8)
        weight=np.append(weight,tmp,axis=2)
    weight=weight.swapaxes(1,2)
    weight=weight.swapaxes(2,3)
    return weight

def generate_weight_1x1(weight):
    ifmch=weight.shape[1]
    ofmch=weight.shape[0]
    weight=weight.reshape([ofmch,ifmch])
    weight_zero=np.zeros([ofmch,ifmch,9]).astype(np.int8)
    weight_zero[:,:,4]=weight
    weight=weight_zero
    weight=weight.reshape([-1,8,ifmch,9])
    weight=weight.swapaxes(1,2)
    weight=weight.swapaxes(2,3)
    return weight

def generate_weight_detect(weight,ofmch_t):
    ifmch=weight.shape[1]
    ofmch=weight.shape[0]
    
    weight=weight.reshape([ofmch,ifmch])
    weight_zero=np.zeros([ofmch,ifmch,9]).astype(np.int8)
    weight_zero[:,:,4]=weight
    weight=weight_zero
    
    #print(weight.shape)
    ofmch=weight.shape[0]
    concat=np.zeros([ofmch_t-ofmch,ifmch,9]).astype(np.int8)
    #print(concat.shape)
    weight=np.concatenate((weight,concat),axis=0)
    #print(weight.shape)
    
    weight=weight.reshape([-1,8,ifmch,9])
    weight=weight.swapaxes(1,2)
    weight=weight.swapaxes(2,3)

    #print(weight.shape)
    return weight

# 每个候选框的深度要保证8的倍数，85的话去掉5或者增加3个通道
def generate_weight_detect_8x(weight,ofmch_t):
    ifmch=weight.shape[1]
    ofmch=weight.shape[0]
    
    weight=weight.reshape([ofmch,ifmch])
    zero=np.zeros([3,ifmch]).astype(np.int8)
    weight=np.concatenate((weight[0:85,:],zero,weight[85:170,:],zero,weight[170:256,:],zero),axis=0)
    ofmch=weight.shape[0]
    weight_zero=np.zeros([ofmch,ifmch,9]).astype(np.int8)
    weight_zero[:,:,4]=weight
    weight=weight_zero
    
    #print(weight.shape)
    ofmch=weight.shape[0]
    concat=np.zeros([ofmch_t-ofmch,ifmch,9]).astype(np.int8)
    #print(concat.shape)
    weight=np.concatenate((weight,concat),axis=0)
    #print(weight.shape)
    
    weight=weight.reshape([-1,8,ifmch,9])
    weight=weight.swapaxes(1,2)
    weight=weight.swapaxes(2,3)

    #print(weight.shape)
    return weight

def generate_weight_detect_75to72(weight,ofmch_t):
    """对于voc模型最后一层，将75个滤波器权重减少为72个"""
    index_w = [24,49,74] 
    weight = np.delete(weight, index_w, axis=0)
    ifmch=weight.shape[1]
    ofmch=weight.shape[0]
    
    weight=weight.reshape([ofmch,ifmch])

    # zero=np.zeros([3,ifmch]).astype(np.int8)
    # weight=np.concatenate((weight[0:85,:],zero,weight[85:170,:],zero,weight[170:256,:],zero),axis=0)
    ofmch=weight.shape[0]
    weight_zero=np.zeros([ofmch,ifmch,9]).astype(np.int8)
    weight_zero[:,:,4]=weight
    weight=weight_zero
    
    #print(weight.shape)
    ofmch=weight.shape[0]
    concat=np.zeros([ofmch_t-ofmch,ifmch,9]).astype(np.int8)
    #print(concat.shape)
    weight=np.concatenate((weight,concat),axis=0)
    #print(weight.shape)
    
    weight=weight.reshape([-1,8,ifmch,9])
    weight=weight.swapaxes(1,2)
    weight=weight.swapaxes(2,3)

    #print(weight.shape)
    return weight
    
def generate_bias(b,bscale):
    bias=torch.quantize_per_tensor(b, scale=bscale, zero_point=0,dtype=torch.qint32).int_repr().numpy().astype(np.int64)
    return bias

def generate_leakytbl(act_scale,act_zp,output_scale,output_zp):
    sm=output_scale/act_scale
    lista=np.array(range(256))
    table=[]
    for data in lista:
        if data<output_zp:
            tmp=round((data-output_zp)*sm*0.1+act_zp)
        else:
            tmp=round((data-output_zp)*sm+act_zp)
        if tmp>=255:
            tmp=255
        table.append(tmp)
    table=np.array(table).astype(np.uint64)
    return table

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def generate_sigmoid_table(zp,scale,rp):
    sigmoid_tbl=sigmoid((np.array(range(256))-zp)*scale)
    np.savetxt(rp+'/sigmoid_table.h',sigmoid_tbl,delimiter=',',encoding='utf-8')
    return sigmoid_tbl
    
def yolov3tiny_infer_para_gen(quant_model,ofmch_t,first_chr,rp):
    convindex=[0,2,4,6,8,10,13,14,15]
    convtype= [0,0,0,0,0,0, 0, 1 ,0 ]
    mult_list=[]
    shift_list=[]
    izp_list=[]
    ozp_list=[]
    azp_list=[]

    ascale=quant_model.state_dict()['0.scale'].item()
    azp=quant_model.state_dict()['0.zero_point'].item()
    cnt=0

    for index in convindex:
        iscale=ascale
        izp=azp
        w,b,wscale,ascale,cscale,azp,czp=generate_para_list(quant_model,index)
        oscale,ozp,bscale,mult,shift=generate_quant_para(iscale,izp,wscale,ascale,cscale,azp,czp)
        if convtype[cnt]==1:
            weight=generate_weight_1x1(w)
        else:
            weight=generate_weight(w)
        bias=generate_bias(b,bscale)
        leakytbl=generate_leakytbl(ascale,azp,oscale,ozp)
        mult_list.append(mult)
        shift_list.append(shift)
        izp_list.append(izp)
        ozp_list.append(ozp)
        azp_list.append(azp)
        weight.tofile(rp + '/infer_bin/'+first_chr+'W'+str(cnt)+'.bin')
        bias.tofile(rp + '/infer_bin/'+first_chr+'B'+str(cnt)+'.bin')
        leakytbl.tofile(rp + '/infer_bin/'+first_chr+'R'+str(cnt)+'.bin')
        #print(cnt,index,w.shape,weight.shape)
        cnt=cnt+1

    iscale=ascale
    izp=azp
    oscale=quant_model.state_dict()['1.model.20.m.1.scale'].item()
    ozp=quant_model.state_dict()['1.model.20.m.1.zero_point'].item()
    wscale=torch.q_scale(quant_model.state_dict()['1.model.20.m.1.weight'])
    m=iscale*wscale/oscale
    base,expr=math.frexp(m)
    mult=round(base*(2**15))
    shift=-expr

    w=quant_model.state_dict()['1.model.20.m.1.weight'].int_repr().numpy().astype(np.int8)
    b=quant_model.state_dict()['1.model.20.m.1.bias']
    bscale=iscale * wscale
    bias=torch.quantize_per_tensor(b, scale=bscale, zero_point=0,dtype=torch.qint32).int_repr().numpy().astype(np.int64)
    # zero=np.zeros([3]).astype(np.int64)
    # bias=np.concatenate((bias[0:85],zero,bias[85:170],zero,bias[170:256],zero))
    index_bias = [24,49,74] 
    bias = np.delete(bias, index_bias)
    print(w.shape)
    weight=generate_weight_detect_75to72(w,ofmch_t)
    # if(first_chr=='Y'):
    #     weight=generate_weight_detect_8x(w,ofmch_t)
    # else:
    #     weight=generate_weight_detect(w,ofmch_t)
    # weight=generate_weight_detect(w,ofmch_t)
    leakytbl=np.array(range(256)).astype(np.uint64)
    sigmoid_scale=quant_model.state_dict()['1.model.20.m.1.scale'].item()
    sigmoid_zp=quant_model.state_dict()['1.model.20.m.1.zero_point'].item()
    print("sigmoid_scale=%.20f" % sigmoid_scale)
    print("sigmoid_zp=%d" % sigmoid_zp)
    sigmoid_tbl=generate_sigmoid_table(sigmoid_zp,sigmoid_scale,rp)
    
    mult_list.append(mult)
    shift_list.append(shift)
    izp_list.append(izp)
    ozp_list.append(ozp)
    azp_list.append(ozp)

    weight.tofile(rp + '/infer_bin/'+first_chr+'W'+str(cnt)+'.bin')
    bias.tofile(rp + '/infer_bin/'+first_chr+'B'+str(cnt)+'.bin')
    leakytbl.tofile(rp + '/infer_bin/'+first_chr+'R'+str(cnt)+'.bin')
    sigmoid_tbl.astype(np.float32).tofile(rp + '/infer_bin/'+first_chr+'SG.bin')
    
    
    sel_in_list =[5,4,3,2,1,0,0,0,0,0];
    pool_list   =[1,1,1,1,1,1,0,0,0,0];
    stride_list =[0,0,0,0,0,1,0,0,0,0];
    ifm_list    =[8,16,32,64,128,256,512,1024,256,512];
    
    ofm_list=[16,32,64,128,256,512,1024,256,512,ofmch_t]
    config_list=ifm_list+ofm_list+mult_list+shift_list+izp_list+ozp_list+azp_list+sel_in_list+pool_list+stride_list
    config_list=np.array(config_list).astype(np.uint32).flatten()
    config_list.astype(np.uint32).tofile(rp + '/infer_bin/'+first_chr+'CG'+'.bin')
    
    print('mult:',mult_list)
    print('shift:',shift_list)
    print('zpin:',izp_list)
    print('zpout:',ozp_list)
    print('zpact:',azp_list)
    # print('list:',config_list)
    return config_list
    

model_name = 'voc_retrain' # coco_origin | coco128_retrain | voc_retrain
relative_path = './models_files/' + model_name

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


float_model,dataloader_iter=load_model_data('./models/yolov3-tiny_voc.yaml',relative_path + '/train/weights/best.pt',416,False) # voc模型
quant_model=load_quant_model(relative_path + '/train/weights/best.pt', relative_path + '/yolov3tiny_quant.pth')
yolov3tiny_infer_para_gen(quant_model,72,'Y',relative_path)

#detect
# view_img = check_imshow()
# cudnn.benchmark = True  # set True to speed up constant image size inference
# dataset = LoadStreams('0', img_size=416, stride=32)
# state_dict = torch.load('./yolov3tiny_facemask_quant.pth')

# data = cv2.imread('data\images\zidane.jpg')
# data = cv2.imread('data/images/bus.jpg')
# data = cv2.imread('data/images/test_1.jpg')
data = cv2.imread('data/images/test_1.jpg')


data,rate,d = letterbox(data,(416, 416),32)
# cv2.imshow('input_image', data)
# cv2.waitKey(0) 
data = np.expand_dims(data.transpose(2,0,1),axis = 0)
data = np.expand_dims(data,axis = 0)
data = torch.from_numpy(data)
quant_model_evaluate_show(data,quant_model)
