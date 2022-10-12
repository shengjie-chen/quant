import serial  #导入串口模块
import time    #导入时间模块

try:
    #打开串口，并且获得串口对象
    MyCom = serial.Serial("com6",115200,timeout=50)
    #判断是否打开成功
    if (MyCom.isOpen()==True):
        print("串口已经打开！")
except Exception as exc:
    print("串口打开异常:",exc)

'''
    我的串口通信
'''
data_ccc = 0
pof=b'40000000'

while True:
    time.sleep(0.1)
    buffer_size = MyCom.in_waiting
    if buffer_size:
       # print("收到",str(buffer_size),"个:  ")
        data = MyCom.read_all()
        #将接受到的数据进行分割处理并打印出来
        data = data[:-2]
        print(data.decode('utf-8', errors='ignore'))
        #设置循环结束条件 
        data_ccc = data_ccc+1
        if data_ccc >= 100:
            break
    else:	# UART发送
        MyCom.write(pof)
   
    
MyCom.close()  

