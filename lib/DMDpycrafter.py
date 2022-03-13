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

        filter = hid.HidDeviceFilter(vendor_id = vendor_num, product_id = product_num)
        devices = filter.get_devices()
        self.dev = devices[0]
    
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

                self.dev.set_raw_data(buffer)
                self.dev.send()
        
        else: # command length is larger than 65 (inculding report ID) bytes
            for i in range(65-len(buffer)):
                buffer.append(data[i])

            self.dev.set_raw_data(buffer) # 1st transfer
            self.dev.send()

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

