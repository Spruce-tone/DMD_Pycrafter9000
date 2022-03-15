'''
copied from this repository
https://github.com/e841018/ERLE
@author Ashu
'''

# encode image of shape (n<=24, 1080, 1920) with Enhanced Run-Length Encoding (ERLE) 
# described in http://www.ti.com/lit/pdf/dlpu018

import numpy as np
import struct
from typing import List
from utils import CustomLogger

logger = CustomLogger().info_logger

pack32be = struct.Struct('>I').pack  # uint32 big endian

WIDTH = int(2560/2)
HEIGHT = 1600

def get_header():
    '''
    generate header defined in section 2.4.2
    '''
    header = bytearray(0)
    # signature
    header += bytearray([0x53, 0x70, 0x6c, 0x64])
    # width
    header += bytearray([WIDTH % 256, WIDTH//256])
    # height
    header += bytearray([HEIGHT % 256, HEIGHT//256])
    # number of bytes, will be overwritten later
    header += bytearray(4)
    # reserved
    header += bytearray([0xff]*8)
    # background color (BB GG RR 00)
    header += bytearray(4)
    # reserved
    header.append(0)
    # compression, 0=Uncompressed, 1=RLE, 2=Enhanced RLE
    header.append(2)
    # reserved
    header.append(1)
    header += bytearray(21)
    return header

header_template = get_header()


def merge(images):
    '''
    merge up to 24 binary images into a single 24-bit image, each pixel is an uint32 of format 0x00BBGGRR
    '''
    image32 = np.zeros((HEIGHT, WIDTH), dtype=np.uint32)
    n_img = len(images)
    batches = [8]*(n_img//8)
    logger.debug(f'batches : {batches}')
    if n_img % 8:
        batches.append(n_img % 8)
    for idx, batch_size in enumerate(batches):
        image8 = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        for j in range(batch_size):
            image8 += images[idx*8+j]*(1 << j)
        image32 += image8*(1 << (idx*8))
    logger.debug(f'image32 | shape : {image32.shape} | dtype : {image32.dtype}')
    return image32


def bgr(pixel):
    '''
    convert an uint32 pixel into [B, G, R] bytes
    '''
    return pack32be(pixel)[1:4]


def enc128(num):
    '''
    encode num (up to 32767) into 1 or 2 bytes
    '''
    return bytearray([(num & 0x7f) | 0x80, num >> 7]) if num >= 128 else bytearray([num])


def run_len(row, idx):
    '''
    find the length of the longest run starting from idx in row
    '''
    stride = 128
    length = len(row)
    j = idx
    while j < length and row[j]:
        if j % stride == 0 and np.all(row[j:j+stride]):
            j += min(stride, length-j)
        else:
            j += 1
    return j-idx


def encode_row(row, same_prev):
    '''
    encode a row of length 1920 with the format described in section 2.4.3.2
    '''
    # same_prev = np.zeros(1920, dtype=bool) if i==0 else image[i]==image[i-1]
    # bool array indicating if same as next element, shape = (width - 1, )
    same = np.logical_not(np.diff(row))
    # same as previous row or same as next element, shape = (width -1 , )
    same_either = np.logical_or(same_prev[:WIDTH-1], same)
    logger.debug(f'same shape  [{same.shape}] | same prev shape [{same_prev.shape}]')
    
    j = 0
    compressed = bytearray(0)
    while j < WIDTH:

        # copy n pixels from previous line
        if same_prev[j]:
            r = run_len(same_prev, j+1) + 1
            j += r
            compressed += b'\x00\x01' + enc128(r)

        # repeat single pixel n times
        elif j < (WIDTH-1) and same[j]:
            r = run_len(same, j+1) + 2
            j += r
            compressed += enc128(r) + bgr(row[j-1])

        # single uncompressed pixel
        elif j > (WIDTH-3) or same_either[j+1]:
            compressed += b'\x01' + bgr(row[j])
            j += 1

        # multiple uncompressed pixels
        else:
            j_start = j
            pixels = bgr(row[j]) + bgr(row[j+1])
            j += 2
            while j == (WIDTH-1) or not same_either[j]:
                pixels += bgr(row[j])
                j += 1
            compressed += b'\x00' + enc128(j-j_start) + pixels

    return compressed + b'\x00\x00'


def encode(images):
    '''
    encode image with the format described in section 2.4.3.2.1
    '''
    # header
    encoded = bytearray(header_template)

    # uint32 array, shape = (1080, 1920)
    image = merge(images)

    # image content
    for i in range(HEIGHT):
        # bool array indicating if same as previous row, shape = (1920, )
        same_prev = np.zeros(WIDTH, dtype=bool) if i == 0 else image[i] == image[i-1]
        encoded += encode_row(image[i], same_prev)

    # end of image
    encoded += b'\x00\x01\x00'

    # pad to 4-byte boundary
    encoded += bytearray((-len(encoded)) % 4)

    # overwrite number of bytes in header
    # uint32 little endian, offset=8
    struct.pack_into('<I', encoded, 8, len(encoded))

    return encoded, len(encoded)

def convlen(num: int,length: int) -> str:
    '''
    converts a number into a bit string of given length

    Input args:
        num: int
            number to be converted
        length: int
            length of output string
    
    Return:
        bin_num: str
            bit string
    '''
    bin_num=bin(num)[2:]
    padding=length-len(bin_num)
    bin_num='0'*padding+bin_num
    return bin_num

def bitstobytes(num: str) -> List[int]:
    '''
    converts a bit string into a given number of bytes

    Input args:
        num: str
            number to be converted ex) '0001'
    
    Return:
        bytelist: List[int]
            converted byte string in list. each number in the list is of 1 byte 
            ex) [1, 0] when input is '0001'
    '''
    bytelist=[]
    if len(num)%8!=0:
        padding=8-len(num)%8
        num='0'*padding+num
    for i in range(len(num)//8):
        bytelist.append(int(num[8*i:8*(i+1)],2))

    bytelist.reverse() # [MSB, LSB] -> [LSB, MSB]
    return bytelist

def new_encode(image):
    """
    I have rewritten the encoding function to make it clearer and straightforward.
    Besides, I have deleted the condition for which the function remains trapped
    in an infinite loop for some hadamard pattern. Everything seems to work fine.
    """

## header creation
    bytecount=48    
    bitstring=[]

    bitstring.append(0x53)
    bitstring.append(0x70)
    bitstring.append(0x6c)
    bitstring.append(0x64)
    
    width=convlen(WIDTH,16)
    width=bitstobytes(width)
    for i in width:
        bitstring.append(i)

    height=convlen(HEIGHT,16)
    height=bitstobytes(height)
    for i in height:
        bitstring.append(i)


    total=convlen(0,32)
    total=bitstobytes(total)
    for i in total:
        bitstring.append(i)        

    for i in range(8):
        bitstring.append(0xff)

    for i in range(4):    ## black curtain
        bitstring.append(0x00)

    bitstring.append(0x00)

    bitstring.append(0x02) ## enhanced rle

    bitstring.append(0x01)

    for i in range(21):
        bitstring.append(0x00)

    n=0
    i=0
    j=0

    while i <HEIGHT:

        while j <WIDTH:

            if i>0:
                if np.all(image[i,j,:]==image[i-1,j,:]):
                    while j<WIDTH and np.all(image[i,j,:]==image[i-1,j,:]):
                        n=n+1
                        j=j+1
                        
    
                    bitstring.append(0x00)
                    bitstring.append(0x01)
                    bytecount+=2
                    
                    if n>=128:
                        byte1=(n & 0x7f)|0x80
                        byte2=(n >> 7)
                        bitstring.append(byte1)
                        bitstring.append(byte2)
                        bytecount+=2
                        
                    else:
                        bitstring.append(n)
                        bytecount+=1
                    n=0

            
                else:
                    if j < WIDTH-1: #1919 since I compare j and j+1 pixel
                        if np.all(image[i,j,:]==image[i,j+1,:]):
                            n=n+1
                    
                            while j<WIDTH-1 and np.all(image[i,j,:]==image[i,j+1,:]):
                                n=n+1
                                j=j+1
                            if n>=128:
                                byte1=(n & 0x7f)|0x80
                                byte2=(n >> 7)
                                bitstring.append(byte1)
                                bitstring.append(byte2)
                                bytecount+=2
                                
                            else:
                                bitstring.append(n)
                                bytecount+=1
        
                            bitstring.append(image[i,j-1,0])
                            bitstring.append(image[i,j-1,1])
                            bitstring.append(image[i,j-1,2])
                            bytecount+=3
                            
                            j=j+1
                            n=0
                        elif j > WIDTH-3 or np.all(image[i,j+1,:]==image[i,j+2,:]) or np.all(image[i,j+1,:]==image[i-1,j+1,:]):
                            bitstring.append(0x01)
                            bytecount+=1
                            bitstring.append(image[i,j,0])
                            bitstring.append(image[i,j,1])
                            bitstring.append(image[i,j,2])
                            bytecount+=3
                            
                            j=j+1
                            n=0
                        else:
                            bitstring.append(0x00)
                            bytecount+=1
    
                            toappend=[]
    
                            while j<WIDTH-1 and np.any(image[i,j,:]!=image[i,j+1,:]):

                                """
                                I've moved the j<1919 condition as first condition since sometimes it
                                tries to read image array at wrong index.
                                """
                                n=n+1
                                toappend.append(image[i,j,0])
                                toappend.append(image[i,j,1])
                                toappend.append(image[i,j,2])
                                j=j+1
                                
                            if n>=128:
                                byte1=(n & 0x7f)|0x80
                                byte2=(n >> 7)
                                bitstring.append(byte1)
                                bitstring.append(byte2)
                                bytecount+=2
    
                            else:
                                bitstring.append(n)
                                bytecount+=1
    
    
                            for k in toappend:
                                bitstring.append(k)
                                bytecount+=1
                            #j=j+1
                            n=0                           
                    elif j == WIDTH-1:
                        
                        bitstring.append(0x01)
                        bytecount+=1
                        bitstring.append(image[i,j,0])
                        bitstring.append(image[i,j,1])
                        bitstring.append(image[i,j,2])
                        bytecount+=3
                        
                        j=j+1
                        n=0
            else:
                
                if j < WIDTH-1: #1919 since I compare j and j+1 pixel
                    logger.debug(f'asdfas {i} {j}')
                    if np.all(image[i,j,:]==image[i,j+1,:]):
                        n=n+1
                
                        while j<WIDTH-1 and np.all(image[i,j,:]==image[i,j+1,:]):
                            n=n+1
                            j=j+1
                        if n>=128:
                            byte1=(n & 0x7f)|0x80
                            byte2=(n >> 7)
                            bitstring.append(byte1)
                            bitstring.append(byte2)
                            bytecount+=2
                            
                        else:
                            bitstring.append(n)
                            bytecount+=1
    
                        bitstring.append(image[i,j-1,0])
                        bitstring.append(image[i,j-1,1])
                        bitstring.append(image[i,j-1,2])
                        bytecount+=3
                        
                        j=j+1
                        n=0
                    elif j > WIDTH-3 or np.all(image[i,j+1,:]==image[i,j+2,:]) or np.all(image[i,j+1,:]==image[i-1,j+1,:]):
                        bitstring.append(0x01)
                        bytecount+=1
                        bitstring.append(image[i,j,0])
                        bitstring.append(image[i,j,1])
                        bitstring.append(image[i,j,2])
                        bytecount+=3
                        
                        j=j+1
                        n=0
                    else:
                        bitstring.append(0x00)
                        bytecount+=1

                        toappend=[]

                        while j<WIDTH-1 and np.any(image[i,j,:]!=image[i,j+1,:]):

                            """
                            I've moved the j<1919 condition as first condition since sometimes it
                            tries to read image array at wrong index.
                            """
                            n=n+1
                            toappend.append(image[i,j,0])
                            toappend.append(image[i,j,1])
                            toappend.append(image[i,j,2])
                            j=j+1
                            
                        if n>=128:
                            byte1=(n & 0x7f)|0x80
                            byte2=(n >> 7)
                            bitstring.append(byte1)
                            bitstring.append(byte2)
                            bytecount+=2

                        else:
                            bitstring.append(n)
                            bytecount+=1


                        for k in toappend:
                            bitstring.append(k)
                            bytecount+=1
                        #j=j+1
                        n=0                           
                elif j == WIDTH-1:
                    
                    bitstring.append(0x01)
                    bytecount+=1
                    bitstring.append(image[i,j,0])
                    bitstring.append(image[i,j,1])
                    bitstring.append(image[i,j,2])
                    bytecount+=3
                    
                    j=j+1
                    n=0
    
        j=0
        i=i+1
        bitstring.append(0x00)
        bitstring.append(0x00)
        bytecount+=2
    bitstring.append(0x00)
    bitstring.append(0x01)
    bitstring.append(0x00)
    bytecount+=3


    while (bytecount)%4!=0:
        bitstring.append(0x00)
        bytecount+=1        

    size=bytecount

    print (size)

    total=convlen(size,32)
    total=bitstobytes(total)
    for i in range(len(total)):
        bitstring[i+8]=total[i]    
    
    return bitstring, bytecount