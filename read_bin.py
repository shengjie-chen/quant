import struct
import os

def ReadFile():
    filepath='./infer_bin/FB0.bin'
    binfile = open(filepath, 'rb') #打开二进制文件
    size = os.path.getsize(filepath) #获得文件大小
    print(size)
    for i in range(size):
        data = binfile.read(1) #每次输出一个字节
        data = struct.unpack('b',data)
        print(data)
    binfile.close()
if __name__ == '__main__':
	ReadFile()
