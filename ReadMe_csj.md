该文件夹为Python模型训练与量化的工程

一些零碎的说明：
在github上面搜索：ultralytics_yolov3能够找到该project的出处

根目录有yolov3和yolov3tiny的官方模型

如果要训练新模型，运行根目录的tran.py脚本，例如

python train.py --img 640 --batch 16 --epochs 100 --data data/facemask.yaml --weights yolov3-tiny.pt --nosave --cache --cfg models/yolov3-tiny-facemask.yaml

如果要用电脑的摄像头去测试一个模型，运行根目录的detect.py脚本，例如

python detect.py --img 416 --source 0 --weights yolov3tiny_facemask.pt

## dir

### /data:
    .yaml文件为各种数据集的信息
    script是几个数据集的获取脚本
    images是默认使用的几张照片

### /models：
    包含了模型函数文件和三种模型的.yaml默认配置文件

### /models_files:
    包括基于yolov3tiny的口罩识别和头盔识别两个模型权重，以及v3和v3tiny的默认模型权重，以.pt为后缀；还有_quant.pth后缀代表的是INT8量化后的模型。

    口罩识别与头盔识别的数据集都来自kaggle

    口罩识别数据集：https://www.kaggle.com/andrewmvd/face-mask-detection

    头盔识别数据集：https://www.kaggle.com/savanagrawal/helmet-detection-yolov3

### /utils：
    包含模型使用的各种函数
    
### /weights：
    包含一个用于下载官方模型权重的脚本

## files

**train**

/train.py: 模型进行训练的脚本

**test**

/test.py: 模型的测试效果脚本

**detect**

/detect.py: 模型的检测使用脚本

/quant_detect.py: 使用训练后静态量化方法实现的量化权重检测脚本，同时将量化模型导出

~~/quant_detect_mask.py:使用训练后静态量化方法实现的量化权重检测脚本，用于识别口罩~~

**val**

/val.py: 在自定义数据集上验证训练的模型准确性

**quant**

/yolov3tiny_quant.py: 将浮点模型使用训练后静态量化方法进行量化并保存


## 数据集

个人测试可用数据集目前有coco128，coco



