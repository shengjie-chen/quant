from pickle import NONE
import serial
from time import sleep

# def recv(serial):
#     num = 0
#     while (num < 20):
#         data = serial.read().hex()
#         print(data)
#     # if data == '':
#     # 	continue
#     # else:
#     # 	break
#         sleep(0.02)
#         num = num + 1

def uartRecv():
    se = serial.Serial('COM6', 115200, timeout=0.5)
    if se.isOpen():
        print("serial open success")
    else:
        print("serial open failed")

    # addr = str(0x35000000).encode("utf-8")
    
    se.write(0x40)
    se.write(0x00)
    se.write(0x00)
    se.write(0x00)
    # for i in range(0, 1):
        # se.write(h'40000000')      
        # data = recv(se)
    # recv(se)
    data = se.read(10).hex()
    print(data)

        # print(data)  # str
    se.close


if __name__ == '__main__':
	uartRecv()