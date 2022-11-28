import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, de_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume
import warnings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


logger = logging.getLogger(__name__)


def train_one_epoch(opt, model, optimizer, device,
                    epoch,  # 当前epoch数
                    epochs,  # 总epoch数
                    pbar,  # 数据集loader的pbar类型
                    nb,  # 数据集的batch数
                    nw,  # 热身训练的迭代次数，至少1000
                    batch_size, nbs,  # nominal batch size 更新权重的bs
                    imgsz,
                    scaler,  # 通过torch1.6自带的api设置混合精度训练
                    plots, save_dir, compute_loss,  cuda, lf,
                    accumulate,
                    gs, mloss,
                    warmup=False  # 是否热身训练
                    ):
    # batch -------------------------------------------------------------
    for i, (imgs, targets, paths, _) in pbar:
        # number integrated batches (since train start) 计算迭代的次数iteration
        ni = i + nb * epoch
        # uint8 to float32, 0-255 to 0.0-1.0
        imgs = imgs.to(device, non_blocking=True).float() / 255.0

        # Warmup 热身训练(前nw次迭代) 在前nw次迭代中，根据以下方式选取accumulate和学习率
        if warmup:
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(
                    ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(
                        ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(
                            ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

        # Multi-scale 设置多尺度训练，从imgsz * 0.5, imgsz * 1.5 + gs随机选取尺寸
        if opt.multi_scale:
            sz = random.randrange(imgsz * 0.5, imgsz *
                                  1.5 + gs) // gs * gs  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                # new shape (stretched to gs-multiple)
                ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]
                imgs = F.interpolate(
                    imgs, size=ns, mode='bilinear', align_corners=False)

        # Forward
        with amp.autocast(enabled=cuda):
            pred = model(imgs)  # forward # 前向传播
            # loss scaled by batch_size  # 计算损失，包括分类损失，objectness损失，框的回归损失  # loss为总损失值，loss_items为一个元组，包含分类损失，objectness损失，框的回归损失和总损失
            loss, loss_items = compute_loss(pred, targets.to(device))

        # Backward 反向传播
        scaler.scale(loss).backward()

        # Optimize 模型反向传播accumulate次之后再根据累积的梯度更新一次参数
        if ni % accumulate == 0:
            scaler.step(optimizer)  # optimizer.step
            scaler.update()
            optimizer.zero_grad()

        # Print
        # 打印显存，进行的轮次，损失，target的数量和图片的size等信息
        mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
        mem = '%.3gG' % (torch.cuda.memory_reserved() /
                         1E9 if torch.cuda.is_available() else 0)  # (GB)
        s = ('%10s' * 2 + '%10.4g' * 6) % (
            '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
        pbar.set_description(s)  # 进度条显示以上信息

        # Plot 将前三次迭代batch的标签框在图片上画出来并保存
        if plots and ni < 3:
            f = save_dir / f'train_batch{ni}.jpg'  # filename
            Thread(target=plot_images, args=(
                imgs, targets, paths, f), daemon=True).start()
            if tb_writer:
                tb_writer.add_graph(torch.jit.trace(de_parallel(
                    model), imgs, strict=False), [])  # model graph
                # tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
    return s, mloss


def train(hyp, opt, device, tb_writer=None):
    """
    训练日志包括：权重、tensorboard文件、超参数hyp、设置的训练参数opt(也就是epochs,batch_size等),result.txt
    result.txt包括: 占GPU内存、训练集的GIOU loss, objectness loss, classification loss, 总loss, 
    targets的数量, 输入图片分辨率, 准确率TP/(TP+FP),召回率TP/P ; 
    测试集的mAP50, mAP@0.5:0.95, GIOU loss, objectness loss, classification loss.
    还会保存batch<3的ground truth
    """
    logger.info(colorstr('hyperparameters: ') +
                ', '.join(f'{k}={v}' for k, v in hyp.items()))  # 加粗打印超参数信息

    # Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)

    # Configure
    epochs = opt.epochs
    batch_size = opt.batch_size
    weights = opt.weights
    nbs = 64  # nominal batch size
    # accumulate loss before optimizing
    accumulate = max(round(nbs / batch_size), 1)

    init_seeds()  # random seed
    with open(opt.data) as f:  # 加载数据配置信息
        data_dict = yaml.safe_load(f)  # data dict
    train_path = data_dict['train']
    test_path = data_dict['val']
    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    hyp['weight_decay'] *= batch_size * \
        accumulate / nbs  # scale weight_decay 权值衰减
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    names = ['item'] if opt.single_cls and len(
        data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (
        len(names), nc, opt.data)  # check
    is_coco = opt.data.endswith('coco.yaml') and nc == 80  # COCO dataset
    plots = True  # create plots
    cuda = device.type != 'cpu'

    # Model
    pretrained = weights.endswith('.pt') or weights.endswith(".pth")
    if pretrained:
        ckpt = torch.load(weights, map_location=device)  # load checkpoint

        # Initialize model
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get(
            'anchors')).to(device)  # create

        exclude = ['anchor'] if (opt.cfg or hyp.get(
            'anchors')) else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        # intersect 似乎是将有序字典变为普通字典
        state_dict = intersect_dicts(
            state_dict, model.state_dict(), exclude=exclude)
        model.load_state_dict(state_dict, strict=False)  # load
        logger.info('Transferred %g/%g items from %s' % (len(state_dict),
                    len(model.state_dict()), weights))  # report 少加载两个键值对(anchors,anchor_grid)
    else:
        model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get(
            'anchors')).to(device)  # create

    # Freeze
    freeze = []  # parameter names to freeze (full or partial)
    for k, v in state_dict.items():
        # if k.startswith('model'):# 冻结所有层
        if opt.freeze_mode == 1:
            if not k.startswith('model.20'):  # 冻结20层之前的所有层
                freeze.append(k)
        elif opt.freeze_mode == 2:
            if not (k.startswith('model.20') or k.startswith('model.15') or k.startswith('model.19')):  # 不冻结最后三层
                freeze.append(k)

    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            logger.info('freezing %s' % k)
            v.requires_grad = False

    # Optimizer  选用优化器
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=hyp['lr0'], momentum=hyp['momentum'],
                          weight_decay=hyp["weight_decay"], nesterov=True)

    # 这里为余弦退火方式进行衰减
    lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    start_epoch, best_fitness = 0, 0.0
    del ckpt, state_dict

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    # number of detection layers (used for scaling hyp['obj'])
    nl = model.model[-1].nl
    # verify imgsz are gs-multiples
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]

    # Trainloader 创建训练集dataloader
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=-1,
                                            world_size=1, workers=opt.workers,
                                            image_weights=False, quad=False, prefix=colorstr('train: '))
    # max label class 获取标签中最大的类别值，并于类别数作比较   如果大于类别数则表示有问题
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (
        mlc, nc, opt.data, nc - 1)

    testloader = create_dataloader(test_path, imgsz_test, batch_size * 2, gs, opt,  # testloader 创建测试集dataloader
                                   hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
                                   world_size=1, workers=opt.workers,
                                   pad=0.5, prefix=colorstr('val: '))[0]

    # plot class
    # 将所有样本的标签拼接到一起shape为(total, 5)，统计后做可视化
    labels = np.concatenate(dataset.labels, 0)
    c = torch.tensor(labels[:, 0])  # classes  获得所有样本的类别
    if plots:
        # 根据上面的统计对所有样本的类别，中心点xy位置，长宽wh做可视化
        plot_labels(labels, names, save_dir)
        if tb_writer:
            tb_writer.add_histogram('classes', c, 0)

    # Anchors
    """
    计算默认锚点anchor与数据集标签框的长宽比值
    标签的长h宽w与anchor的长h_a宽w_a的比值, 即h/h_a, w/w_a都要在(1/hyp['anchor_t'], hyp['anchor_t'])是可以接受的
    如果标签框满足上面条件的数量小于总数的99%，则根据k-mean算法聚类新的锚点anchor
    """
    if not opt.noautoanchor:
        check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
    model.half().float()  # pre-reduce anchor precision

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    # scale to classes and layers 根据自己数据集的类别数设置分类损失的系数
    hyp['cls'] *= nc / 80. * 3. / nl
    # scale to image size and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.names = names

    # Start training
    t0 = time.time()
    # number of warmup iterations, max(3 epochs, 1k iterations)
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)
    # limit warmup to < 1/2 of training
    nw = min(nw, (epochs - start_epoch) / 2 * nb)
    maps = np.zeros(nc)  # mAP per class 初始化mAP和results
    # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    results = (0, 0, 0, 0, 0, 0, 0)
    scheduler.last_epoch = start_epoch
    scaler = amp.GradScaler(enabled=cuda)  # 通过torch1.6自带的api设置混合精度训练
    compute_loss = ComputeLoss(model)  # init loss class
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')

    # epoch ------------------------------------------------------------------
    for epoch in range(start_epoch, epochs):
        model.train()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # mean losses 初始化训练时打印的平均损失信息

        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem',
                    'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        # batch -------------------------------------------------------------
        s, mloss = train_one_epoch(opt, model, optimizer, device, epoch, epochs, pbar, nb, nw, batch_size, nbs, imgsz,
                                   scaler, plots, save_dir, compute_loss,  cuda, lf, accumulate, gs, mloss, warmup=opt.warmup)

        # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------

        # Scheduler 进行学习率衰减
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        # mAP # 更新EMA的属性  添加include的属性
        final_epoch = epoch + 1 == epochs
        if not opt.notest or final_epoch:  # Calculate mAP  对测试集进行测试，计算mAP等指标 测试时使用的是EMA模型
            results, maps, times = test.test(data_dict,
                                             batch_size=batch_size * 2,
                                             imgsz=imgsz_test,
                                             model=model,
                                             single_cls=opt.single_cls,
                                             dataloader=testloader,
                                             save_dir=save_dir,
                                             save_json=is_coco and final_epoch,
                                             verbose=nc < 50 and final_epoch,
                                             plots=plots and final_epoch,
                                             wandb_logger=None,
                                             compute_loss=compute_loss,
                                             is_coco=is_coco)

        # Write 将指标写入result.txt
        with open(results_file, 'a') as f:
            # append metrics, val_loss
            f.write(s + '%10.4g' * 7 % results + '\n')

        # Log
        tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                'x/lr0', 'x/lr1', 'x/lr2']  # params
        for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
            if tb_writer:
                tb_writer.add_scalar(tag, x, epoch)  # tensorboard

        # Update best mAP
        # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        fi = fitness(np.array(results).reshape(1, -1))
        if fi > best_fitness:
            best_fitness = fi

        # Save model
        if (not opt.nosave) or (final_epoch):  # if save
            ckpt = {'epoch': epoch,
                    'best_fitness': best_fitness,
                    'training_results': results_file.read_text(),
                    'model': deepcopy(de_parallel(model)).half(),
                    'optimizer': optimizer.state_dict()}

            # Save last, best and delete
            torch.save(ckpt, last)
            if best_fitness == fi:
                torch.save(ckpt, best)
            del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    logger.info(
        f'{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.\n')
    if plots:
        plot_results(save_dir=save_dir)  # save as results.png

        if is_coco:  # COCO dataset
            for m in [last, best] if best.exists() else [last]:  # speed, mAP tests
                results, _, _ = test.test(opt.data,
                                          batch_size=batch_size * 2,
                                          imgsz=imgsz_test,
                                          conf_thres=0.001,
                                          iou_thres=0.7,
                                          model=attempt_load(m, device).half(),
                                          single_cls=opt.single_cls,
                                          dataloader=testloader,
                                          save_dir=save_dir,
                                          save_json=True,
                                          plots=False,
                                          is_coco=is_coco)

        # Strip optimizers
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='yolov3.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str,
                        default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument(
        '--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16,
                        help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int,
                        default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true',
                        help='rectangular training')
    # parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true',
                        help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true',
                        help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true',
                        help='disable autoanchor check')
    # parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    # parser.add_argument('--bucket', type=str, default='', help='gsutil bucket') # 超参进化相关？
    parser.add_argument('--cache-images', action='store_true',
                        help='cache images for faster training')
    # parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true',
                        help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true',
                        help='train multi-class data as single-class')
    # parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    # parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    # parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=8,
                        help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train',
                        help='save to project/name')
    # parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--warmup', action='store_true', help='warm up train')
    parser.add_argument('--freeze_mode', type=int, default=0,
                        help='support 0/1/2')
    # parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # parser.add_argument('--quad', action='store_true', help='quad dataloader')
    # parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    # parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    # parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    # parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    # parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    # parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    opt = parser.parse_args()
    warnings.filterwarnings('ignore')

    # set_logging()
    check_requirements(exclude=('pycocotools', 'thop'))

    opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(
        opt.cfg), check_file(opt.hyp)  # check files 检查文件是否存在
    assert len(opt.cfg) or len(
        opt.weights), 'either --cfg or --weights must be specified'
    # extend to 2 sizes (train, test)
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name))
    os.mkdir(opt.save_dir)

    # set_logging()
    log_file = opt.save_dir + '/train.log'
    os.mknod(log_file)
    logger.setLevel(level=logging.DEBUG)

    file_formatter = logging.Formatter(
        '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    stream_formatter = logging.Formatter('%(message)s')

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(file_formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(stream_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)  # load hyps

    # Train
    logger.info(opt)
    tb_writer = None  # init loggers
    prefix = colorstr('tensorboard: ')
    logger.info(
        f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
    tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
    train(hyp, opt, device, tb_writer)
