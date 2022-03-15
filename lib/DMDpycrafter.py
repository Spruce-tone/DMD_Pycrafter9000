from typing import List
# import pywinusb.hid as hid
import hid
import time
import numpy as np
import sys
sys.path.append('./lib')
from erle import encode, new_encode
from utils import CustomLogger

logger = CustomLogger().info_logger

WIDTH = int(2560/2)
HEIGHT = 1600

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


class DMDhid():
    def __init__(self):
        #usb Vendor ID
        vendor_num = 0x0451
        #usb Product ID
        product_num = 0xc900

        hid.enumerate()
        self.device = hid.device()
        self.device.open(0x0451, 0xc900)
        time.sleep(0.5)
        # filter = hid.HidDeviceFilter(vendor_id = vendor_num, product_id = product_num)
        # devices = filter.get_devices()
        # self.dev = devices[1]
        # self.dev.open()

        # self.out_report = self.dev.find_output_reports()[0]

       
    def read_handler(data):
        return data   

    def command(self,mode,sequencebyte,com1,com2,data=None):
        '''
        _summary_

        Parameters
        ----------
        mode : _type_
            _description_
        sequencebyte : _type_
            _description_
        com1 : _type_
            _description_
        com2 : _type_
            _description_
        data : _type_, optional
            _description_, by default None


        description
        ----------
        command for USB 1.1 HID protocol
        Reporte ID  | Header Bytes                                          | Payload Bytes
                    | [Flag byte, Sequence byte, Length LSB, Length MSB]    | [USB command [LSB,    MSB   ], Data ......]
        [Byte 0   ] | [Byte 1   , Byte 2       , Byte 3    , Byte 4    ]    | [             Byte 5, Byte 6, ..... Byte N]

        Flag byte
        [Read/write , Reply      , Error    , Reserved  , Destination]
        [bit 7      , bit 6      , bit 5    , bit 4-3   , bit 2-0    ]
        
        1. Report ID byte : always set to 0
        2. Flag byte
            - bit 2-0 : set to 0x0 for regular DLPC900 operation
            - bit 6   : 0x1, host wants a reply from the device
                        0x0, host doesn't want a reply
            - bit 5   : 0, No errors 
                        1, command not found or command failed
        3. Sequence byte: The sequence byte can be a rolling counter. It is used primarily when the host wants a
           response from the DLPC900. The DLPC900 will respond with the same sequence byte that the host
           sent. The host can then match the sequence byte from the command it sent with the sequence byte
           from the DLPC900 response.
        4. Length
            - Two bytes in length, this denotes the number of data bytes in the Payload only.
            ex) '00010011' = 19
                Most significant bit (MSB) = '0001' = 1
                Least significant bit (LSB) = '0011' = 3
                therefor, in the command,
                [3, 1] = [Length LSB (Byte 3), Length MSB (Byte 4)]
        5. USB command
            - two bytes USB command [LSB (Byte 5), MSB (Byte 5)]


        DLPC900 internal command buffer has a maximum of 512 bytes.
        The HID protocol is limited to 64 byte transfers in both directions (65 bytes including Report iD). 
        Therefore, commands that are larger than 64 bytes require multiple transfers.
        
        ex)
        case I) command length <= 65 bytes
        Report ID - Header bytes - Paload Bytes

        case II) command length > 65 bytes
        if the command length is 83 bytes
        first transfer  : Report ID (byte 0)  - Header bytes (byte 1-4) - Paload Bytes (bytes 5-64)
        second transfer : Report ID (byte 65) - Paload Bytes (bytes 66-83)
        '''
        buffer = []

        flagstring=''
        if mode=='r':
            flagstring+='1'
        else:
            flagstring+='0'        
        flagstring+='1000000' # host wants a reply from device (bit 7 = 1)
        buffer.append(0x00) # Report ID [Byte 0]
        buffer.append(bitstobytes(flagstring)[0]) # Flag byte [Byte 1]
        buffer.append(sequencebyte) # Sequence byte [Byte 2]
        # Because the USB command length is 2 bytes in Payload, 2 is added to the data length 
        payload_len_byte=bitstobytes(convlen(len(data)+2,16)) # payload length in byte = usb command (2 bytes) + data length (len(data))
        buffer.append(payload_len_byte[0]) # payload length LSB [Byte 3]
        buffer.append(payload_len_byte[1]) # payload length MSB [Byte 4]
        buffer.append(com2) # usb command LSB [Byte 5]
        buffer.append(com1) # usb command MSB [Byte 6]

        
        if len(buffer) + len(data) <= 65: # maximum 65 bytes including Report ID can be sent per transmission
            # add data to buffer
            for i in data:
                buffer.append(i) 

            # fill the remaining buffer with 0x00 (1 byte)
            for i in range(65 - len(buffer)):
                buffer.append(0x00)

            self.device.write(buffer)
        
        else: # command length is larger than 65 (inculding report ID) bytes
            for i in range(65-len(buffer)):
                buffer.append(data[i])

            self.device.write(buffer)

            buffer = [0x00]

            j=0
            while j<len(data)-58:
                buffer.append(data[j+58])
                j=j+1
                if j%64==0: #we need 64 instead of 65
                    self.device.write(buffer)

                    buffer = [0x00]

            if j%64!=0:

                while j%64!=0:
                    buffer.append(0x00)
                    j=j+1
                    
                self.device.write(buffer)

    def checkforerrors(self):
        """
        This part needs to be checked
        """
        self.ans = self.device.read(1)
        self.flag = convlen(self.ans[0], 8)

        if self.flag[2]=="1":
            print("An error occurred! --> ", self.ans)
            self.command('r',0x22,0x01,0x00,[]) # read error code
            self.error = self.device.read(1)

            self.command('r',0x22,0x01,0x01,[]) # read error description
            self.response = self.device.read(128) 

## function printing all of the dlp answer

    def readreply(self):
        for i in self.ans:
            print (hex(i))

## functions for idle mode activation

    def idle_on(self):
        self.command('w',0x00,0x02,0x01,[int('00000001',2)])
        self.checkforerrors()

    def idle_off(self):
        self.command('w',0x00,0x02,0x01,[int('00000000',2)])
        self.checkforerrors()

## functions for power management

    def standby(self):
        self.command('w',0x00,0x02,0x00,[int('00000001',2)])
        self.checkforerrors()

    def wakeup(self):
        self.command('w',0x00,0x02,0x00,[int('00000000',2)])
        self.checkforerrors()

    def reset(self):
        self.command('w',0x00,0x02,0x00,[int('00000010',2)])
        self.checkforerrors()

## test write and read operations, as reported in the dlpc900 programmer's guide

    def testread(self):
        self.command('r',0xff,0x11,0x00,[])
        self.readreply()

    def testwrite(self):
        self.command('w',0x22,0x11,0x00,[0xff,0x01,0xff,0x01,0xff,0x01])
        self.checkforerrors()

## some self explaining functions

    def changemode(self,mode):
        self.command('w',0x00,0x1a,0x1b,[mode])
        self.checkforerrors()

    def startsequence(self):
        self.command('w',0x00,0x1a,0x24,[2])
        self.checkforerrors()

    def pausesequence(self):
        self.command('w',0x00,0x1a,0x24,[1])
        self.checkforerrors()

    def stopsequence(self):
        self.command('w',0x00,0x1a,0x24,[0])
        self.checkforerrors()


    def configurelut(self,imgnum,repeatnum):
        img=convlen(imgnum,11)
        repeat=convlen(repeatnum,32)

        string=repeat+'00000'+img

        bytes=bitstobytes(string)

        self.command('w',0x00,0x1a,0x31,bytes)
        self.checkforerrors()
        

    def definepattern(self,index,exposure,bitdepth,color,triggerin,darktime,triggerout,patind,bitpos):
        payload=[]
        index=convlen(index,16)
        index=bitstobytes(index)
        for i in range(len(index)):
            payload.append(index[i])

        exposure=convlen(exposure,24)
        exposure=bitstobytes(exposure)
        for i in range(len(exposure)):
            payload.append(exposure[i])
        optionsbyte=''
        optionsbyte+='1'
        bitdepth=convlen(bitdepth-1,3)
        optionsbyte=bitdepth+optionsbyte
        optionsbyte=color+optionsbyte
        if triggerin:
            optionsbyte='1'+optionsbyte
        else:
            optionsbyte='0'+optionsbyte

        payload.append(bitstobytes(optionsbyte)[0])

        darktime=convlen(darktime,24)
        darktime=bitstobytes(darktime)
        for i in range(len(darktime)):
            payload.append(darktime[i])

        triggerout=convlen(triggerout,8)
        triggerout=bitstobytes(triggerout)
        payload.append(triggerout[0])

        patind=convlen(patind,11)
        bitpos=convlen(bitpos,5)
        lastbits=bitpos+patind
        lastbits=bitstobytes(lastbits)
        for i in range(len(lastbits)):
            payload.append(lastbits[i])



        self.command('w',0x00,0x1a,0x34,payload)
        self.checkforerrors()
        


    def setbmp(self,index,size, controller='master'):
        payload=[]

        index=convlen(index,5)
        index='0'*11+index
        index=bitstobytes(index)
        for i in range(len(index)):
            payload.append(index[i]) 


        total=convlen(size,32)
        total=bitstobytes(total)
        for i in range(len(total)):
            payload.append(total[i])         
        
        if controller=='master':
            self.command('w',0x00,0x1a,0x2a,payload)
        elif controller=='slave':
            self.command('w',0x00,0x1a,0x2c,payload)
            
        self.checkforerrors()

## bmp loading function, divided in 56 bytes packages
## max  hid package size=64, flag bytes=4, usb command bytes=2
## size of package description bytes=2. 64-4-2-2=56

    def bmpload(self,image,size, controller='master'):

        packnum=size//504+1

        counter=0

        for i in range(packnum):
            if i %100==0:
                print (i,packnum)
            payload=[]
            if i<packnum-1:
                leng=convlen(504,16)
                bits=504
            else:
                leng=convlen(size%504,16)
                bits=size%504
            leng=bitstobytes(leng)
            for j in range(2):
                payload.append(leng[j])
            for j in range(bits):
                payload.append(image[counter])
                counter+=1
            
            if controller=='master':
                self.command('w',0x11,0x1a,0x2b,payload)
            elif controller=='slave':
                self.command('w',0x11,0x1a,0x2d,payload)


            self.checkforerrors()

    def defsequence(self,images,exp,ti,dt,to,rep):

        self.stopsequence()

        master_arr=[]
        slave_arr = []

        for img in images:
            master_arr.append(img[:, :WIDTH])
            slave_arr.append(img[:, WIDTH:])
        logger.debug(f'master_arr length : {len(master_arr)}')
        logger.debug(f'slave_arr length : {len(slave_arr)}')

##        arr.append(np.ones((1080,1920),dtype='uint8'))

        num=len(master_arr)

        maserter_encodedimages=[]
        slave_encodedimages=[]
        maseter_sizes=[]
        salve_sizes=[]

        for i in range((num-1)//24+1):
            print ('merging...')
            if i<((num-1)//24):
                master_imagedata=master_arr[i*24:(i+1)*24]
                slave_imagedata=slave_arr[i*24:(i+1)*24]
                logger.debug(f'{i} < ((num-1)//24) [{((num-1)//24)}]')
                logger.debug(f'imgdata length | Master [{len(master_imagedata)}] | Slave [{len(slave_imagedata)}]')
            else:
                master_imagedata=master_arr[i*24:]
                slave_imagedata=slave_arr[i*24:]
                logger.debug(f'{i} >= ((num-1)//24) [{((num-1)//24)}]')
                logger.debug(f'imgdata length | Master [{len(master_imagedata)}] | Slave [{len(slave_imagedata)}]')

            print ('encoding...')
            master_imagedata = mergeimages(master_imagedata)
            slave_imagedata = mergeimages(slave_imagedata)

            master_imagedata, master_size=new_encode(master_imagedata)
            slave_imagedata, slave_size=new_encode(slave_imagedata)
            logger.debug(f'imgdata length | Master [{len(master_imagedata)}] | Slave [{len(slave_imagedata)}]')

            logger.debug(f'Master Imagedata, size : [{len(master_imagedata)}], [{master_size}]')
            logger.debug(f'Slave Imagedata, size : [{len(slave_imagedata)}], [{slave_size}]')

            maserter_encodedimages.append(master_imagedata)
            slave_encodedimages.append(slave_imagedata)

            logger.debug(f'Master encodedimages length : {len(maserter_encodedimages)}')
            logger.debug(f'Slave encodedimages length : {len(slave_encodedimages)}')

            maseter_sizes.append(master_size)
            salve_sizes.append(slave_size)
            logger.debug(f'Master sizes length : {len(maseter_sizes)}')
            logger.debug(f'Slave sizes length : {len(salve_sizes)}')

            if i<((num-1)//24):
                for j in range(i*24,(i+1)*24):
                    self.definepattern(j,exp[j],1,'111',ti[j],dt[j],to[j],i,j-i*24)
            else:
                for j in range(i*24,num):
                    self.definepattern(j,exp[j],1,'111',ti[j],dt[j],to[j],i,j-i*24)

        self.configurelut(num,rep)

        for i in range((num-1)//24+1):
        
            self.setbmp((num-1)//24-i, maseter_sizes[(num-1)//24-i], controller='master')
            self.setbmp((num-1)//24-i, salve_sizes[(num-1)//24-i], controller='slave')

            print ('uploading...')
            self.bmpload(maserter_encodedimages[(num-1)//24-i],maseter_sizes[(num-1)//24-i], controller='master')
            self.bmpload(slave_encodedimages[(num-1)//24-i],salve_sizes[(num-1)//24-i], controller='slave')


def mergeimages(images):
    """
    function that encodes a 8 bit numpy array matrix as a enhanced run length encoded string of bits
    """
    mergedimage=np.zeros((HEIGHT, WIDTH,3),dtype='uint8') #put this as np.uint8?

    for i in range(len(images)):
        
        if i<8:
            mergedimage[:,:,2]=mergedimage[:,:,2]+images[i]*(2**i) #with the multiplication, the 8 bit pixel contains the info abopu all the 8 images (if we are in binary...)

        if i>7 and i<16:
            mergedimage[:,:,1]=mergedimage[:,:,1]+images[i]*(2**(i-8))

        if i>15 and i<24: #maybe 24 because in RGB you have 8*3
            mergedimage[:,:,0]=mergedimage[:,:,0]+images[i]*(2**(i-16))
            
    return mergedimage