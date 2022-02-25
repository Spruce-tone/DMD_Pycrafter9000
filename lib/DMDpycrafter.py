from typing import List
from pywinusb import hid



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
    padding=length-len(num)
    bin_num='0'*padding+bin_num
    return bin_num

def bitstobytes(num) -> List[int]:
    '''
    converts a bit string into a given number of bytes

    Input args:
        num: int
            number to be converted
    
    Return:
        bytelist: List[int]
            converted byte string in list. each number in the list is of 1 bytes 
    '''
    bytelist=[]
    if len(num)%8!=0:
        padding=8-len(num)%8
        num='0'*padding+num
    for i in range(len(num)//8):
        bytelist.append(int(num[8*i:8*(i+1)],2))

    bytelist.reverse()
    return bytelist


class DMDhid():
    def __init__(self):
        #usb Vendor ID
        vendor_num = 0x0451
        #usb Product ID
        product_num = 0xc900

        filter = hid.HidDeviceFilter(vendor_id = vendor_num, product_id = product_num)
        devices = filter.get_devices()
        self.dev = devices[0]
    
    def command(self,mode,sequencebyte,com1,com2,data=None):
        '''
        command for USB 1.1 HID protocol
        Reporte ID  | Header Bytes                                          | Payload Bytes
                    | [Flag byte, Sequence byte, Length LSB, Length MSB]    | [USB command [LSB, MSB], Data..]
        [Byte 0   ] | [Byte 1   , Byte 2       , Byte 3    , Byte 4    ]    | [Bytes 5 ... N                 ]

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
        3. Length
            - Two bytes in length, this denotes the number of data bytes in the Payload only.
            ex) '00010011' = 19
                Most significant bit (MSB) = '0001' = 1
                Least significant bit (LSB) = '0011' = 3
                therefor, in the command,
                [3, 1] = [Length LSB (byte 3), Length MSB (byte 4)]


        DLPC900 internal command buffer has a maximum of 512 bytes.
        The HID protocol is limited to 64 byte transfers in both directions. Therefore, commands that are larger
        than 64 bytes require multiple transfers.
        
        ex)
        case I) command length < 65 bytes
        Report ID - Header bytes - Paload Bytes

        case II) command length >= 65 bytes
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
        buffer.append(bitstobytes(flagstring)[0])
        buffer.append(sequencebyte)
        payload_len_byte=bitstobytes(convlen(len(data)+2,16)) # Because the USB command length is 2 bytes in Payload, 2 is added to the data length 
        buffer.append(payload_len_byte[0]) # payload length LSB
        buffer.append(payload_len_byte[1]) # payload length MSB
        buffer.append(com2) # usb command LSB
        buffer.append(com1) # usb command MSB