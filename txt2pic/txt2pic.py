import struct
import os
import numpy as np
import cv2

"""将txt文件中16进制数读出来,组成图片并显示"""

def ReadTxt2Array(img):
    filepath='./txt2pic/image_418x258.txt'
    file = open(filepath)               #打开txt文件
    for y in range(258):
        for x in range(418):
            for c in range(8):
                data = file.read(2)             #每次输出一个字节
                data = int(data,16)
                # print(data)
                file.read(1)
                if c < 3 :
                    img[y][x][c] = data
    file.close()

def ReadTxt2Array_real(img):    # 0B 1G 2R
    filepath='./txt2pic/rawData0x36000000.txt'
    file = open(filepath)               #打开txt文件
    for y in range(258):
        for x in range(418):
            for c in range(8):
                data = file.read(2)             #每次输出一个字节
                if(data[0] == '\n'):
                    data = data[1] + file.read(1)
                data = 2 * int(data,16)
                if c < 3 :
                    img[y][x][c] = data
    file.close()

if __name__ == '__main__':
    img = np.zeros((258,418,3),np.uint8)
    ReadTxt2Array_real(img)
    img = img[:, :, [2, 1, 0]]
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # img = cv2.imread('test_1.jpg') #0B 1G 2R
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

